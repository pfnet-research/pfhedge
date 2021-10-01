from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim import Optimizer

# error: Skipping analyzing "tqdm": found module but no type hints or library stubs
from tqdm import tqdm  # type: ignore

from pfhedge._utils.hook import save_prev_output
from pfhedge._utils.lazy import has_lazy
from pfhedge._utils.operations import ensemble_mean
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.features import FeatureList
from pfhedge.features._base import Feature
from pfhedge.instruments.base import Instrument
from pfhedge.instruments.derivative.base import Derivative
from pfhedge.instruments.primary.base import Primary
from pfhedge.nn.functional import terminal_value

from .loss import EntropicRiskMeasure
from .loss import HedgeLoss


class Hedger(Module):
    """Module to hedge and price derivatives.

    References:
        - Buehler, H., Gonon, L., Teichmann, J. and Wood, B., 2019.
          Deep hedging. Quantitative Finance, 19(8), pp.1271-1291.
          [arXiv:`1802.03042 <https://arxiv.org/abs/1802.03042>`_ [q-fin]]

    Args:
        model (torch.nn.Module): Hedging model to compute the hedge ratio at the
            next time step from the input features at the current time step.
            The input and output shapes should be :math:`(N, F)` and
            :math:`(N, H)` respectively, where
            :math:`N` stands for the number simulated paths of the asset prices and
            :math:`F` is the number of input features (``len(inputs)``), and
            :math:`H` is the number of hedging instruments.
        inputs (list[str|Feature]): List of the names of the input features that
            will be fed to the model.
            See ``list(map(str, pfhedge.features.FEATURES))`` for valid options.
        criterion (HedgeLoss, default=EntropicRiskMeasure()):
            Loss function to minimize by hedging.
            Default: :class:`pfhedge.nn.EntropicRiskMeasure()` .

    Shape:
        - input: :math:`(N, F)` where
          :math:`N` is the number of simulated paths and
          :math:`F` is the number of input features.
        - output: :math:`(N, H)` where
          :math:`H` is the number of hedging instruments.

    Examples:

        A hedger that uses Black-Scholes' delta hedging strategy.
        See :class:`pfhedge.nn.BlackScholes` for details of the module.

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> from pfhedge.nn import BlackScholes
        >>> from pfhedge.nn import Hedger
        >>>
        >>> derivative = EuropeanOption(BrownianStock(cost=1e-4))
        >>> model = BlackScholes(derivative)
        >>> hedger = Hedger(model, model.inputs())
        >>> hedger
        Hedger(
          inputs=['log_moneyness', 'time_to_maturity', 'volatility']
          (model): BSEuropeanOption(strike=1.)
          (criterion): EntropicRiskMeasure()
        )

        A hedger that uses Whalley-Wilmott's no-transaction-band strategy.
        See :class:`pfhedge.nn.WhalleyWilmott` for details of the module.

        >>> from pfhedge.nn import WhalleyWilmott
        >>>
        >>> model = WhalleyWilmott(derivative)
        >>> hedger = Hedger(model, model.inputs())
        >>> hedger
        Hedger(
          inputs=['log_moneyness', 'time_to_maturity', 'volatility', 'prev_hedge']
          (model): WhalleyWilmott(
            (bs): BSEuropeanOption(strike=1.)
            (clamp): Clamp()
          )
          (criterion): EntropicRiskMeasure()
        )

        A hedger that takes naked positions (never hedge at all).
        See :class:`pfhedge.nn.Naked` for details of the module.

        >>> from pfhedge.nn import Naked
        >>>
        >>> hedger = Hedger(Naked(), ["empty"])

        A hedger represented by a neural network (Deep Hedging).
        See :class:`pfhedge.nn.MultiLayerPerceptron` for details of the module.

        >>> from pfhedge.nn import MultiLayerPerceptron
        >>>
        >>> model = MultiLayerPerceptron()
        >>> hedger = Hedger(model, ["moneyness", "time_to_maturity", "volatility"])
        >>> _ = hedger.compute_pnl(derivative, n_paths=1)  # Lazily materialize
        >>> hedger
        Hedger(
          inputs=['moneyness', 'time_to_maturity', 'volatility']
          (model): MultiLayerPerceptron(
            (0): Linear(in_features=3, out_features=32, bias=True)
            (1): ReLU()
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): ReLU()
            (4): Linear(in_features=32, out_features=32, bias=True)
            (5): ReLU()
            (6): Linear(in_features=32, out_features=32, bias=True)
            (7): ReLU()
            (8): Linear(in_features=32, out_features=1, bias=True)
            (9): Identity()
          )
          (criterion): EntropicRiskMeasure()
        )
        >>> history = hedger.fit(derivative, n_paths=1, n_epochs=1, verbose=False)
        >>> hedger.price(derivative)
        tensor(...)

        It is possible to hedge a derivative with another listed derivative by
        ``Derivative.list()`` method.

        >>> from pfhedge.instruments import LookbackOption
        >>> from pfhedge.nn import BlackScholes
        >>>
        >>> pricer = lambda derivative: BlackScholes(derivative).price(
        ...     log_moneyness=derivative.log_moneyness(),
        ...     time_to_maturity=derivative.time_to_maturity(),
        ...     volatility=derivative.ul().volatility)
        >>>
        >>> stock = BrownianStock()
        >>> hedging_instrument = EuropeanOption(stock, maturity=5/250)
        >>> hedging_instrument.list(pricer, cost=1e-4)
        >>> derivative = LookbackOption(stock)
        >>>
        >>> hedger = Hedger(
        ...     MultiLayerPerceptron(),
        ...     inputs=["moneyness", "time_to_maturity", "volatility"])
        >>> _ = hedger.fit(
        ...     derivative,
        ...     hedge=[hedging_instrument],
        ...     n_paths=1,
        ...     n_epochs=1,
        ...     verbose=False)
        >>> hedger.price(derivative)
        tensor(...)

        Hedging a derivative with multiple instruments.

        >>> from pfhedge.instruments import HestonStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> from pfhedge.instruments import VarianceSwap
        >>> from pfhedge.nn import BlackScholes
        >>>
        >>> _ = torch.manual_seed(42)
        >>> stock = HestonStock(cost=1e-4)
        >>> option = EuropeanOption(stock)
        >>> varswap = VarianceSwap(stock)
        >>> pricer = lambda varswap: varswap.ul().variance - varswap.strike
        >>> varswap.list(pricer, cost=1e-4)
        >>> hedger = Hedger(
        ...     MultiLayerPerceptron(3, 2),
        ...     inputs=["moneyness", "time_to_maturity", "volatility"])
        >>> hedger.price(option, hedge=[stock, varswap], n_paths=2)
        tensor(...)
    """

    inputs: FeatureList

    def __init__(
        self,
        model: Module,
        inputs: List[Union[str, Feature]],
        criterion: HedgeLoss = EntropicRiskMeasure(),
    ):
        super().__init__()

        self.model = model
        self.inputs = FeatureList(inputs)
        self.criterion = criterion

        self.register_forward_hook(save_prev_output)

    def forward(self, input: Tensor) -> Tensor:
        """Returns the outout of ``self.model``.

        The output represents the hedge ratio at the next time step.
        """
        return self.model(input)

    def extra_repr(self) -> str:
        return "inputs=" + str(self.inputs)

    def get_input(self, derivative: Derivative, time_step: Optional[int]) -> Tensor:
        """Returns the input tensor to the model at the given time step.

        Note:
            This method assumes that a derivative is already registered to
            the features. If self has not yet hedged a derivative,
            run a placeholder computation
            ``_ = self.compute_pnl(derivative, n_paths=1)``
            before calling this method.

        Args:
            derivative (Derivative): The derivative used for getting the input.
            time_step (int, optional): The time step to get the input tensor.
                If ``None`` an input tensor for all time steps is returned.

        Shape:
            - Output: :math:`(N, T, F)` where
              :math:`N` is the number of paths,
              :math:`T` is the number of time steps, and
              :math:`F` is the number of input features.
              If ``time_step`` is specified, :math:`T = 1`.

        Returns:
            torch.Tensor

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import Naked
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> derivative.simulate()
            >>> hedger = Hedger(Naked(), ["time_to_maturity", "volatility"])
            >>> _ = hedger.compute_pnl(derivative, n_paths=1)  # Materialize features
            >>> hedger.get_input(derivative, 0)
            tensor([[[0.0800, 0.2000]]])
        """
        return self.inputs.of(derivative=derivative).get(time_step)

    def compute_hedge(
        self, derivative: Derivative, hedge: Optional[List[Instrument]] = None
    ) -> Tensor:
        """Compute the hedge ratio at each time step.
        It assumes that the derivative is already simulated.

        Args:
            derivative (Derivative): The derivative to hedge.
            hedge (Instrument, optional): The hedging instrument.
                If ``None`` (default), use ``derivative.underlier``.

        Shape:
            - Output: :math:`(N, H, T)` where
              :math:`N` is the number of paths,
              :math:`H` is the number of hedging instruments, and
              :math:`T` is the number of time steps.

        Returns:
            torch.Tensor

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import BlackScholes
            >>>
            >>> _ = torch.manual_seed(42)
            >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250)
            >>> derivative.simulate(n_paths=2)
            >>> derivative.ul().spot
            tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906],
                    [1.0000, 0.9919, 0.9976, 1.0009, 1.0076, 1.0179]])
            >>> model = BlackScholes(derivative)
            >>> hedger = Hedger(model, model.inputs())
            >>> hedger.compute_hedge(derivative).squeeze(1)
            tensor([[0.5056, 0.5295, 0.5845, 0.6610, 0.2918, 0.2918],
                    [0.5056, 0.3785, 0.4609, 0.5239, 0.7281, 0.7281]])
        """
        inputs = self.inputs.of(derivative, self)
        hedge = cast(List[Instrument], [derivative.ul()] if hedge is None else hedge)

        # Check that the spot prices of the hedges have the same sizes
        if not all(h.spot.size() == hedge[0].spot.size() for h in hedge):
            raise ValueError("The spot prices of the hedges must have the same size")

        (n_paths, n_steps), n_hedges = hedge[0].spot.size(), len(hedge)
        if inputs.is_state_dependent():
            zeros = hedge[0].spot.new_zeros((n_paths, 1, n_hedges))
            save_prev_output(self, input=None, output=zeros)
            outputs = []
            for time_step in range(n_steps - 1):
                input = inputs.get(time_step)  # (N, T=1, F)
                outputs.append(self(input))  # (N, T=1, H)
            outputs.append(outputs[-1])
            output = torch.cat(outputs, dim=-2)  # (N, T, H)
        else:
            # If all features are state-independent, compute the output at all
            # time steps at once, which would be faster.
            input = inputs.get(None)  # (N, T, F)
            output = self(input)  # (N, T, H)
            # This maintains consistency with the previous implementations.
            # In previous implementation for loop is computed for 0...T-2 and
            # the last time step is not included.
            output[..., -1, :] = output[..., -2, :]

        output = output.transpose(-1, -2)  # (N, H, T)

        return output

    def compute_pnl(
        self,
        derivative: Derivative,
        hedge: Optional[List[Instrument]] = None,
        n_paths: int = 1000,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> Tensor:
        """Returns the terminal portfolio value after hedging a given derivative.

        This method simulates the derivative, computes the hedge ratio, and
        computes the terminal portfolio value.
        See :func:`pfhedge.nn.functional.terminal_value` for the expression of the
        terminal portyol value after hedging a derivative.

        Args:
            derivative (Derivative): The derivative to hedge.
            hedge (list[Instrument], optional): The hedging instruments.
                If ``None`` (default), use ``[derivative.underlier]``.
            n_paths (int, default=1000): The number of simulated price paths of the
                underlying instrument.
            init_state (tuple[torch.Tensor | float], optional): The initial state of
                the underlying instrument of the derivative.
                If ``None`` (default), it uses the default value.

        Shape:
            - Output: :math:`(N)` where
              :math:`N` is the number of paths.

        Returns:
            torch.Tensor

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import BlackScholes
            >>> from pfhedge.nn import Hedger
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> model = BlackScholes(derivative)
            >>> hedger = Hedger(model, model.inputs())
            >>> hedger.compute_pnl(derivative, n_paths=2)
            tensor([..., ...])
        """
        derivative.simulate(n_paths=n_paths, init_state=init_state)
        hedge = cast(List[Instrument], [derivative.ul()] if hedge is None else hedge)

        unit = self.compute_hedge(derivative, hedge=hedge)

        output = -derivative.payoff()
        for i, h in enumerate(hedge):
            output += terminal_value(h.spot, unit=unit[:, i, :], cost=h.cost)

        return output

    def compute_loss(
        self,
        derivative: Derivative,
        hedge: Optional[List[Instrument]] = None,
        n_paths: int = 1000,
        n_times: int = 1,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
        enable_grad: bool = True,
    ) -> Tensor:
        """Returns the value of the criterion for the terminal portfolio value
        after hedging a given derivative.

        This method basically computes ``self.criterion(pnl)``
        where ``pnl`` is given by :meth:`compute_pnl`.

        Args:
            derivative (Derivative): The derivative to hedge.
            hedge (list[Instrument], optional): The hedging instruments.
                If ``None`` (default), use ``[derivative.underlier]``.
            n_paths (int, default=1000): The number of simulated price paths of the
                underlying instrument.
            n_times (int, default=1): If ``n_times > 1``, returns the ensemble mean
                of the losses computed through multiple simulations.
            init_state (tuple, optional): The initial price of the underlying
                instrument of the derivative.
                If ``None`` (default), it uses the default value of
                the underlying instrument.
            enable_grad (bool, default=True): Context-manager that sets gradient
                calculation to on or off.

        Shape:
            - Output: :math:`()`

        Returns:
            torch.Tensor

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import BlackScholes
            >>> from pfhedge.nn import Hedger
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> model = BlackScholes(derivative)
            >>> hedger = Hedger(model, model.inputs())
            >>> hedger.compute_loss(derivative, n_paths=2)
            tensor(...)
        """
        with torch.set_grad_enabled(enable_grad):
            loss = lambda: self.criterion(
                self.compute_pnl(
                    derivative, hedge=hedge, n_paths=n_paths, init_state=init_state
                )
            )
            mean_loss = ensemble_mean(loss, n_times=n_times)

        return mean_loss

    def _configure_optimizer(
        self,
        derivative: Derivative,
        optimizer: Union[Optimizer, Callable[..., Optimizer]],
    ) -> Optimizer:
        if not isinstance(optimizer, Optimizer):
            if has_lazy(self):
                # Run a placeholder forward to initialize lazy parameters
                _ = self.compute_pnl(derivative, n_paths=1)
            # If we use `if issubclass(optimizer, Optimizer)` here, mypy thinks that
            # optimizer is Optimizer rather than its subclass (e.g. Adam)
            # and complains that the required parameter default is missing.
            if Optimizer in getattr(optimizer, "__mro__", []):
                optimizer = cast(Optimizer, optimizer(self.model.parameters()))
            else:
                raise TypeError("optimizer is not an Optimizer type")
        return optimizer

    def fit(
        self,
        derivative: Derivative,
        hedge: Optional[List[Instrument]] = None,
        n_epochs: int = 100,
        n_paths: int = 1000,
        n_times: int = 1,
        optimizer: Union[Optimizer, Callable[..., Optimizer]] = Adam,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
        verbose: bool = True,
        validation: bool = True,
    ) -> Optional[List[float]]:
        """Fit the hedging model to hedge a given derivative.

        The training is performed so that the hedger minimizes ``criterion(pnl)``
        where ``pnl`` is given by :meth:`compute_pnl`.

        It returns the training history, that is,
        validation loss after each simulation.

        Args:
            derivative (Derivative): The derivative to hedge.
            hedge (list[Instrument], optional): The hedging instruments.
                If ``None`` (default), use ``[derivative.underlier]``.
            n_epochs (int, default=100): Number of Monte-Carlo simulations.
            n_paths (int, default=1000): The number of simulated price paths of the
                underlying instrument.
            n_times (int, default=1): If ``n_times > 1``, returns the ensemble mean of
                the losses computed through multiple simulations.
            optimizer (torch.optim.Optimizer, default=Adam): The optimizer algorithm
                to use.  It can be an instance or a class of
                :class:`torch.optim.Optimizer`.
            init_state (tuple, optional): The initial price of the underlying
                instrument of the derivative.
                If ``None`` (default), sensible default value is used.
            verbose (bool, default=True): If ``True``, print progress of the training to
                standard output.
            validation (bool, default=True): If ``False``, skip the computation of the
                validation loss and returns ``None``.

        Returns:
            list[float]

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import MultiLayerPerceptron
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> model = MultiLayerPerceptron()
            >>> hedger = Hedger(model, ["moneyness", "time_to_maturity", "volatility"])
            >>> history = hedger.fit(derivative, n_paths=1, n_epochs=1, verbose=False)

            One can use a custom optimizer as follows.

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import MultiLayerPerceptron
            >>> from torch.optim import SGD
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> hedger = Hedger(MultiLayerPerceptron(), ["empty"])
            >>> # Run a placeholder forward to initialize lazy parameters
            >>> _ = hedger.compute_pnl(derivative, n_paths=1)
            >>> _ = hedger.fit(
            ...     derivative,
            ...     optimizer=SGD(hedger.parameters(), lr=0.1),
            ...     n_epochs=1,
            ...     verbose=False)

            One can also pass a class object of an optimizer.
            The optimizer will be initialized as ``Adadelta(hedger.parameters())``.

            >>> from torch.optim import Adadelta
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> hedger = Hedger(MultiLayerPerceptron(), ["empty"])
            >>> _ = hedger.fit(
            ...     derivative,
            ...     optimizer=Adadelta,
            ...     n_epochs=1,
            ...     verbose=False)
        """
        optimizer = self._configure_optimizer(derivative, optimizer)

        def compute_loss(**kwargs) -> Tensor:
            return self.compute_loss(
                derivative,
                hedge=hedge,
                n_paths=n_paths,
                init_state=init_state,
                **kwargs,
            )

        history = []
        progress = tqdm(range(n_epochs), disable=not verbose)
        for _ in progress:
            # Compute training loss and backpropagate
            self.train()
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()

            # Compute validation loss
            if validation:
                self.eval()
                loss = compute_loss(n_times=n_times, enable_grad=False)
                history.append(loss.item())

                progress.desc = "Loss=" + _format_float(float(loss.item()))

        return history if validation else None

    def price(
        self,
        derivative: Derivative,
        hedge: Optional[List[Instrument]] = None,
        n_paths: int = 1000,
        n_times: int = 1,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
        enable_grad: bool = False,
    ) -> Tensor:
        """Evaluate the premium of the given derivative.

        Args:
            derivative (Derivative): The derivative to price.
            hedge (list[Instrument], optional): The hedging instruments.
                If ``None`` (default), use ``[derivative.underlier]``.
            n_paths (int, default=1000): The number of simulated price paths of the
                underlying instrument.
            n_times (int, default=1): If ``n_times > 1``, returns the ensemble mean of
                the losses computed through multiple simulations.
            init_state (tuple, optional): The initial price of the underlying
                instrument of the derivative.
                If ``None`` (default), it uses the default value of
                the underlying instrument.
            enable_grad (bool, default=False): Context-manager that sets gradient
                calculation to on or off.

        Shape:
            - Output: :math:`()`

        Returns:
            torch.Tensor

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>> from pfhedge.nn import BlackScholes
            >>> from pfhedge.nn import Hedger
            >>>
            >>> derivative = EuropeanOption(BrownianStock())
            >>> model = BlackScholes(derivative)
            >>> hedger = Hedger(model, model.inputs())
            >>> hedger.price(derivative, n_paths=2)
            tensor(...)
        """
        with torch.set_grad_enabled(enable_grad):
            # Negative because selling
            pricer = lambda: -self.criterion.cash(
                self.compute_pnl(
                    derivative, hedge=hedge, n_paths=n_paths, init_state=init_state
                )
            )
            mean_price = ensemble_mean(pricer, n_times=n_times)

        return mean_price
