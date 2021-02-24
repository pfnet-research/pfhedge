from ..nn.functional import american_binary_payoff
from ._base import Derivative


class AmericanBinaryOption(Derivative):
    """
    An American binary Option.

    An American binary call option pays an unit amount of cash if and only if
    the maximum of the underlying asset's price until maturity is equal or greater
    than the strike price.

    An American binary put option pays an unit amount of cash if and only if
    the minimum of the underlying asset's price until maturity is equal or smaller
    than the strike price.

    The payoff of an American binary call option is given by:

        payoff = 1 if Max >= K
                 0 otherwise

        Max = The maximum of the underlying asset's price until maturity
        K = The strike price (`strike`) of the option

    The payoff of an American binary put option is given by:

        payoff = 1 if Min <= K
                 0 otherwise

        Min = The minimum of the underlying asset's price until maturity
        K = The strike price (`strike`) of the option

    Parameters
    ----------
    - underlier : Primary
        The underlying instrument of the option.
    - call : bool, default True
        Specify whether the option is call or put.
    - strike : float, default 1.0
        The strike price of the option.
    - maturity : float, default 30 / 365
        The maturity of the option.

    Examples
    --------
    >>> import torch
    >>> from pfhedge.instruments import BrownianStock

    >>> _ = torch.manual_seed(42)
    >>> deriv = AmericanBinaryOption(
    ...     BrownianStock(), maturity=5 / 365, strike=1.01)
    >>> deriv.simulate(n_paths=2)
    >>> deriv.underlier.prices
    tensor([[1.0000, 1.0000],
            [1.0024, 1.0024],
            [0.9906, 1.0004],
            [1.0137, 0.9936],
            [1.0186, 0.9964]])
    >>> deriv.payoff()
    tensor([1., 0.])
    """

    def __init__(self, underlier, call=True, strike=1.0, maturity=30 / 365):
        super().__init__()
        self.underlier = underlier
        self.call = call
        self.strike = strike
        self.maturity = maturity

    def __repr__(self):
        params = [f"{self.underlier.__class__.__name__}(...)"]
        if not self.call:
            params.append(f"call={self.call}")
        if self.strike != 1.0:
            params.append(f"strike={self.strike}")
        params.append(f"maturity={self.maturity:.2e}")

        return self.__class__.__name__ + f"({', '.join(params)})"

    def payoff(self):
        return american_binary_payoff(
            self.underlier.prices, call=self.call, strike=self.strike
        )
