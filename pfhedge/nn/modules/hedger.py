from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam

# error: Skipping analyzing "tqdm": found module but no type hints or library stubs
from tqdm import tqdm  # type: ignore

from pfhedge._utils.hook import save_prev_output
from pfhedge._utils.lazy import has_lazy
from pfhedge._utils.operations import ensemble_mean
from pfhedge._utils.str import _format_float
from pfhedge.features import FeatureList
from pfhedge.features._base import Feature
from pfhedge.instruments.derivative.base import Derivative
from pfhedge.instruments.primary.base import Primary
from pfhedge.nn.functional import terminal_value

from .loss import EntropicRiskMeasure
from .loss import HedgeLoss

TensorOrFloat = Union[Tensor, float]


class Hedger(Module):
    """A :class:`torch.nn.Module` to hedge and price derivatives.

    Args:
        model (torch.nn.Module): Hedging model to compute the hedge ratio at the
            next time step from the input features at the current time step.
            The input and output shapes should be :math:`(N, H_\\text{in})` and
            :math:`(N, 1)` respectively, where :math:`N` stands for the number simulated
            paths of the asset prices and :math:`H_\\text{in}` stands for the number of
            input features (namely, ``len(inputs)``).
        inputs (list[str|Feature]): List of (names of) input features to feed to model.
            See ``list(map(str, pfhedge.features.FEATURES))`` for valid options.
        criterion (HedgeLoss, default=EntropicRiskMeasure()):
            Loss function to minimize by hedging.
            Default: :class:`pfhedge.nn.EntropicRiskMeasure()` .

    Shape:
        - Input: :math:`(N, H_{\\text{in}})` where :math:`H_{\\text{in}}` is
          the number of input features.
        - Output: :math:`(N, 1)`

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
        ...     hedge=hedging_instrument,
        ...     n_paths=1,
        ...     n_epochs=1,
        ...     verbose=False)
        >>> hedger.price(derivative)
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

    def get_input(self, time_step: Optional[int]) -> Tensor:
        """Returns the input tensor to the model at the given time step.

        Note:
            This method assumes that a derivative is already registered to
            the features. If self has not yet hedged a derivative,
            run a placeholder computation
            ``_ = self.compute_pnl(derivative, n_paths=1)``
            before calling this method.

        Args:
            time_step (int, optional): The time step to get the input tensor.
                If ``None`` an input tensor for all time steps is returned.

        Shape:
            - Output: :math:`(N, T, F)` where :math:`N` is the number of paths,
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
            >>> hedger.get_input(0)
            tensor([[[0.0800, 0.2000]]])
        """
        return self.inputs[time_step]

    def compute_hedge(
        self, derivative: Derivative, hedge: Optional[Union[Primary, Derivative]] = None
    ) -> Tensor:
        """Compute the hedge ratio at each time step.
        It assumes that the derivative is already simulated.

        Args:
            derivative (Derivative): The derivative to hedge.
            hedge (Instrument, optional): The hedging instrument.
                If ``None`` (default), use ``derivative.underlier``.

        Shape:
            - Output: :math:`(N, H, T)` where :math:`N` is the number of paths,
              :math:`H = 1` is the number of hedging instruments, and
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
        self.inputs = self.inputs.of(derivative, self)

        hedge = derivative.ul() if hedge is None else hedge
        hedge = cast(Union[Primary, Derivative], hedge)

        if self.inputs.is_state_dependent():
            save_prev_output(
                self, None, torch.zeros_like(hedge.spot[..., :1]).unsqueeze(-1)
            )
            outputs = []
            for time_step in range(hedge.spot.size(-1) - 1):
                input = self.get_input(time_step)  # (N, T=1, F)
                outputs.append(self(input))  # (N, T=1, H=1)
            outputs.append(outputs[-1])
            output = torch.cat(outputs, dim=-2)  # (N, T, H=1)
        else:
            # If all features are state-independent, compute the output at all
            # time steps at once, which would be faster.
            input = self.get_input(None)  # (N, T, F)
            output = self(input)  # (N, T, H=1)
            # This maintains consistency with the previous implementations.
            # In previous implementation for loop is computed for 0...T-2 and
            # the last time step is not included.
            output[..., -1, :] = output[..., -2, :]

        output = output.transpose(-1, -2)  #  (N, H=1, T)

        return output

    def compute_pnl(
        self,
        derivative: Derivative,
        hedge: Optional[Union[Primary, Derivative]] = None,
        n_paths: int = 1000,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
    ) -> Tensor:
        """Returns the profit and loss distribution after hedging.

        A hedger sells the derivative to its customer and obliges to settle the payoff
        at maturity. The dealer hedges the risk of this liability by trading
        the underlying instrument of the derivative based on ``self.model``.
        The resulting profit and loss is obtained by adding up the payoff to the
        customer, capital gains from the underlying asset, and the transaction cost.

        Args:
            derivative (Derivative): The derivative to hedge.
            hedge (Instrument, optional): The hedging instrument.
                If ``None`` (default), use ``derivative.underlier``.
            n_paths (int, default=1000): The number of simulated price paths of the
                underlying instrument.
            init_state (tuple[torch.Tensor | float], optional): The initial state of
                the underlying instrument of the derivative.
                If ``None`` (default), it uses the default value.

        Shape:
            - Output: :math:`(N)`, where :math:`N` is the number of paths.

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
        hedge = derivative.ul() if hedge is None else hedge
        hedge = cast(Union[Primary, Derivative], hedge)

        derivative.simulate(n_paths=n_paths, init_state=init_state)

        unit = self.compute_hedge(derivative, hedge=hedge)  # (N, H=1, T)
        unit = unit.squeeze(-2)  # (N, T)

        return terminal_value(
            hedge.spot, unit=unit, cost=hedge.cost, payoff=derivative.payoff()
        )

    def compute_loss(
        self,
        derivative: Derivative,
        hedge: Optional[Union[Primary, Derivative]] = None,
        n_paths: int = 1000,
        n_times: int = 1,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
        enable_grad: bool = True,
    ) -> Tensor:
        """Returns the loss of the profit and loss distribution after hedging.

        Args:
            derivative (Derivative): The derivative to hedge.
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

    def fit(
        self,
        derivative: Derivative,
        hedge: Optional[Union[Primary, Derivative]] = None,
        n_epochs: int = 100,
        n_paths: int = 1000,
        n_times: int = 1,
        optimizer=Adam,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
        verbose: bool = True,
    ) -> List[float]:
        """Train the hedging model to hedge the given derivative.

        It returns the trade history, that is, validation loss after each simulation.

        Args:
            derivative (Derivative): The derivative to hedge.
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
        if isinstance(optimizer, type):
            if has_lazy(self):
                # Run a placeholder forward to initialize lazy parameters
                _ = self.compute_pnl(derivative, n_paths=1)
            optimizer = optimizer(self.model.parameters())
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("optimizer is not torch.optim.Optimizer")

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
            self.eval()
            loss = compute_loss(n_times=n_times, enable_grad=False)
            history.append(loss.item())

            if verbose:
                progress.desc = "Loss=" + _format_float(float(loss.item()))

        return history

    def price(
        self,
        derivative: Derivative,
        hedge: Optional[Union[Primary, Derivative]] = None,
        n_paths: int = 1000,
        n_times: int = 1,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
        enable_grad: bool = False,
    ) -> Tensor:
        """Evaluate the premium of the given derivative.

        Args:
            derivative (Derivative): The derivative to price.
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
