import torch
from torch.optim import Adam
from tqdm import tqdm

from ._utils.hook import save_prev_output
from ._utils.operations import ensemble_mean
from .features import get_feature
from .nn import EntropicRiskMeasure


class Hedger(torch.nn.Module):
    """
    Module to hedge and price derivatives.

    Parameters
    ----------
    - model : torch.nn.Module
        Hedging model to compute the hedge ratio at the next time step from
        the input features at the current time step.
        The input and output shapes should be `(N, H_in) -> (N, 1)`,
        where `N` stands for the number of Monte Carlo paths of the asset prices
        and `H_in` stands for the number of input features
        (namely, `H_in = len(features)`).
    - features : list[str|Feature]
        List of (names of) features to feed to model.
        See `sorted(pfhedge.features.FEATURES)` for valid options.
    - criterion : HedgeLoss, default `EntropicRiskMeasure()`
        Loss function to minimize by hedging.

    Shape
    -----
    Returns the output of `model`.

    Input : (N, H_in)
        See `features` for input features.
        Here, `H_in` is the number of input features.
    Output : (N, 1)
        The hedge ratio at the next time step.

    Examples
    --------
    >>> from pfhedge.instruments import BrownianStock
    >>> from pfhedge.instruments import EuropeanOption

    >>> deriv = EuropeanOption(BrownianStock(cost=1e-4))

    Black-Scholes' delta hedging strategy.

    >>> from pfhedge.nn import BlackScholes

    >>> model = BlackScholes(deriv)
    >>> hedger = Hedger(model, model.features())
    >>> hedger
    Hedger(
      features=['log_moneyness', 'expiry_time', 'volatility'],
      (model): BSEuropeanOption()
      (criterion): EntropicRiskMeasure()
    )
    >>> _ = torch.manual_seed(42)
    >>> hedger.compute_pnl(deriv, n_paths=5)
    tensor([-0.0209, -0.0257, -0.0224, -0.0174, -0.0202])
    >>> _ = torch.manual_seed(42)
    >>> hedger.price(deriv)
    tensor(0.0223)

    Whalley-Wilmott's no-transaction-band strategy.

    >>> from pfhedge.nn import WhalleyWilmott

    >>> model = WhalleyWilmott(deriv)
    >>> hedger = Hedger(model, model.features())
    >>> hedger
    Hedger(
      features=['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge'],
      (model): WhalleyWilmott(
        (bs): BSEuropeanOption()
        (clamp): Clamp()
      )
      (criterion): EntropicRiskMeasure()
    )
    >>> _ = torch.manual_seed(42)
    >>> hedger.compute_pnl(deriv, n_paths=5)
    tensor([-0.0229, -0.0357, -0.0221, -0.0062, -0.0251])
    >>> _ = torch.manual_seed(42)
    >>> hedger.price(deriv)
    tensor(0.0226)

    A naked position (never hedge at all).

    >>> from pfhedge.nn import Naked

    >>> hedger = Hedger(Naked(), ["zero"])
    >>> _ = torch.manual_seed(42)
    >>> hedger.compute_pnl(deriv, n_paths=5)
    tensor([-0.0259, -0.1227,  0.0000, -0.0090, -0.0541])
    >>> _ = torch.manual_seed(42)
    >>> hedger.price(deriv)
    tensor(0.0246)

    A strategy represented by a neural network (Deep Hedging).

    >>> from pfhedge.nn import MultiLayerPerceptron

    >>> _ = torch.manual_seed(42)
    >>> model = MultiLayerPerceptron()
    >>> hedger = Hedger(model, ["moneyness", "expiry_time", "volatility"])
    >>> _ = hedger.compute_pnl(deriv, n_paths=1)  # lazily derermine in_features
    >>> hedger
    Hedger(
      features=['moneyness', 'expiry_time', 'volatility'],
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
    >>> _ = torch.manual_seed(42)
    >>> history = hedger.fit(deriv, verbose=False, n_paths=1, n_epochs=1)
    >>> _ = torch.manual_seed(42)
    >>> hedger.compute_pnl(deriv, n_paths=5)
    tensor([-0.0277, -0.1312,  0.0053, -0.0096, -0.0578], grad_fn=<SubBackward0>)
    >>> _ = torch.manual_seed(42)
    >>> hedger.price(deriv)
    tensor(0.0249)
    """

    def __init__(self, model, features, criterion=EntropicRiskMeasure()):
        super().__init__()

        self.model = model
        self.features = [get_feature(feature) for feature in features]
        self.criterion = criterion

        # This hook saves the hedger's previous output to an attribute `prev`.
        # The attribute `prev` may be referred to by the feature `PrevHedge`.
        self.register_forward_hook(save_prev_output)

    def forward(self, input):
        return self.model(input)

    def extra_repr(self) -> str:
        params = []
        if not isinstance(self.model, torch.nn.Module):
            params.append(f"model={self.model.__name__},")
        params.append(f"features={[str(f) for f in self.features]},")

        return "\n".join(params)

    def compute_pnl(self, derivative, n_paths=1000, init_price=1.0) -> torch.tensor:
        """
        Returns the profit and loss distribution after hedging.

        A hedger sells the derivative to its customer and obliges to settle the payoff
        at maturity. The dealer hedges the risk of this liability by trading
        the underlying instrument of the derivative based on `model`.
        The resulting profit and loss is obtained by adding up the payoff to the
        customer, capital gains from the underlying asset, and the transaction cost.

        Parameters
        ----------
        - derivative : Derivative
            The derivative to hedge.
        - n_paths : int, default 1000
            The number of simulated price paths of the underlying instrument.
        - init_price : float, default 1.0
            The initial price of the underlying instrument of the derivative.

        Returns
        -------
        profit_and_loss : Tensor, shape (n_paths,)
        """
        self.features = [feature.of(derivative, self) for feature in self.features]

        derivative.simulate(n_paths=n_paths, init_price=init_price)
        cashflow = derivative.underlier.prices[1:] - derivative.underlier.prices[:-1]

        self.prev = torch.zeros_like(derivative.underlier.prices[:1]).reshape(-1)
        pnl = 0

        # Simulate hedging over time.
        n_steps = derivative.underlier.prices.size()[0]
        for i in range(n_steps - 1):
            prev_hedge = self.prev.reshape(-1)

            # Compute the hedge ratio at the next time step.
            hedge = self(torch.cat([f[i] for f in self.features], 1)).reshape(-1)

            # Receive profit and loss from the underlying asset.
            pnl += hedge * cashflow[i]
            # Deduct transactoon cost.
            pnl -= (
                derivative.underlier.cost
                * torch.abs(hedge - prev_hedge)
                * derivative.underlier.prices[i]
            )

        # Settle the derivative's payoff.
        pnl -= derivative.payoff()

        # Delete the attribute `prev` in case a hedger has a feature `PrevHedge` and
        # one calls `compute_pnl` twice.
        # If `prev` is not deleted, `prev` at the last time step in the first call
        # would be referred to by `PrevHedge` at the first time step at the second call,
        # which results in an unexpected output (while we expect zeros).
        if hasattr(self, "prev"):
            delattr(self, "prev")

        return pnl

    def compute_loss(
        self, derivative, n_paths=1000, n_times=1, init_price=1.0, enable_grad=True
    ) -> torch.Tensor:
        """
        Returns the loss of the profit and loss distribution after hedging.

        Parameters
        ----------
        - derivative : Derivative
            The derivative to hedge.
        - n_paths : int, default 1000
            The number of simulated price paths of the underlying instrument.
        - n_times : int, default 1
            If `n_times > 1`, returns the ensemble mean of the losses computed
            through multiple simulations.
        - init_price : float, default 1.0
            The initial price of the underlying instrument of the derivative.
        - enable_grad : bool, default True
            Context-manager that sets gradient calculation to on or off.

        Returns
        -------
        loss : Tensor, shape ()
        """
        with torch.set_grad_enabled(enable_grad):
            loss = lambda: self.criterion(
                self.compute_pnl(derivative, init_price=init_price, n_paths=n_paths)
            )
            mean_loss = ensemble_mean(loss, n_times=n_times)

        return mean_loss

    def fit(
        self,
        derivative,
        n_epochs=100,
        n_paths=1000,
        n_times=1,
        optimizer=Adam,
        init_price=1.0,
        verbose=True,
    ):
        """
        Train the hedging model to hedge the given derivative.

        Parameters
        ----------
        - derivative : Derivative
            The derivative to hedge.
        - n_epochs : int, default 100
            Number of Monte-Carlo simulations.
        - n_paths : int, default 1000
            The number of simulated price paths of the underlying instrument.
        - n_times : int, default 1
            If `n_times > 1`, returns the ensemble mean of the losses computed
            through multiple simulations.
        - optimizer : torch.optim.Optimizer, default Adam
            The optimizer algorithm to use.
            It can be an instance or a class of `torch.optim.Optimizer`.
        - init_price : float, default 1.0
            The initial price of the underlying instrument of the derivative.
        - enable_grad : bool, default True
            Context-manager that sets gradient calculation to on or off.
        - verbose : bool, default True
            If `True`, print progress of the training to standard output.

        Returns
        -------
        history : list[float], len `n_epochs`
            Loss after each simulation.

        Examples
        --------
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> from pfhedge.nn import MultiLayerPerceptron

        Using a custom optimizer:

        >>> from torch.optim import SGD

        >>> derivative = EuropeanOption(BrownianStock())
        >>> hedger = Hedger(MultiLayerPerceptron(), ["zero"])
        >>> # Run a "dummy" forward to initialize lazy parameters
        >>> _ = hedger.compute_pnl(derivative, n_paths=1)
        >>> _ = hedger.fit(
        ...     derivative,
        ...     optimizer=SGD(hedger.model.parameters(), lr=0.1),
        ...     n_epochs=1,
        ...     verbose=False,
        ... )

        You can also pass a class object of an optimizer.

        >>> from torch.optim import Adadelta

        >>> derivative = EuropeanOption(BrownianStock())
        >>> hedger = Hedger(MultiLayerPerceptron(), ["zero"])
        >>> _ = hedger.fit(
        ...     derivative,
        ...     optimizer=Adadelta,
        ...     n_epochs=1,
        ...     verbose=False,
        ... )
        """
        if isinstance(optimizer, type):
            # Run a "dummy" forward to initialize lazy parameters
            _ = self.compute_pnl(derivative, n_paths=1)
            optimizer = optimizer(self.model.parameters())
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("optimizer is not torch.optim.Optimizer")

        compute_loss = lambda **kwargs: self.compute_loss(
            derivative, n_paths=n_paths, init_price=init_price, **kwargs
        )

        history = []
        progress = tqdm(range(n_epochs)) if verbose else range(n_epochs)
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
                progress.desc = f"Loss={loss:.5e}"

        return history

    def price(
        self, derivative, n_paths=1000, n_times=1, init_price=1.0, enable_grad=False
    ) -> torch.Tensor:
        """
        Evaluate the premium of the given derivative.

        Parameters
        ----------
        - derivative : Derivative
            The derivative to price.
        - n_paths : int, default 1000
            The number of simulated price paths of the underlying instrument.
        - n_times : int, default 1
            If `n_times > 1`, returns the ensemble mean of the losses computed
            through multiple simulations.
        - init_price : float, default 1.0
            The initial price of the underlying instrument of the derivative.
        - enable_grad : bool, default False
            Context-manager that sets gradient calculation to on or off.

        Returns
        -------
        price : Tensor, shape ()
        """
        with torch.set_grad_enabled(enable_grad):
            # Negative because selling
            price = lambda: -self.criterion.cash(
                self.compute_pnl(derivative, init_price=init_price, n_paths=n_paths)
            )
            mean_price = ensemble_mean(price, n_times=n_times)

        return mean_price
