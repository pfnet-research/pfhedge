import torch

from ...instruments import AmericanBinaryOption
from ...instruments import EuropeanBinaryOption
from ...instruments import EuropeanOption
from ...instruments import LookbackOption
from .american_binary import BSAmericanBinaryOption
from .european import BSEuropeanOption
from .european_binary import BSEuropeanBinaryOption
from .lookback import BSLookbackOption


class BlackScholes(torch.nn.Module):
    """
    Initialize Black-Scholes formula module from a derivative.
    The `forward` method returns the Black-Scholes delta.

    Parameters
    ----------
    - derivative : Derivative
        The derivative to get the Black-Scholes formula.

    Shape
    -----
    - Input : (N, *, H_in)
        See `features()` for input features.
        Here, `*` means any number of additional dimensions and `H_in` is
        the number of input features..
    - Output : (N, *, 1)
        where all but the last dimension are the same shape as the input.

    Examples
    --------
    >>> import torch
    >>> from pfhedge.instruments import BrownianStock
    >>> from pfhedge.instruments import EuropeanOption
    >>> from pfhedge.instruments import LookbackOption

    >>> deriv = EuropeanOption(BrownianStock())
    >>> m = BlackScholes(deriv)
    >>> m
    BSEuropeanOption()
    >>> m.features()
    ['log_moneyness', 'expiry_time', 'volatility']
    >>> x = torch.tensor([
    ...     [-0.01, 0.1, 0.2],
    ...     [ 0.00, 0.1, 0.2],
    ...     [ 0.01, 0.1, 0.2]])
    >>> m(x)
    tensor([[0.4497],
            [0.5126],
            [0.5752]])

    >>> deriv = LookbackOption(BrownianStock(), strike=1.03)
    >>> m = BlackScholes(deriv)
    >>> m
    BSLookbackOption(strike=1.03)
    >>> m.features()
    ['log_moneyness', 'max_log_moneyness', 'expiry_time', 'volatility']
    >>> x = torch.tensor([
    ...     [-0.01, -0.01, 0.1, 0.2],
    ...     [ 0.00,  0.00, 0.1, 0.2],
    ...     [ 0.01,  0.01, 0.1, 0.2]])
    >>> m(x)
    tensor([[0.9208],
            [1.0515],
            [1.0515]])
    """

    def __init__(self, derivative):
        self.__class__ = {
            EuropeanOption: BSEuropeanOption,
            LookbackOption: BSLookbackOption,
            AmericanBinaryOption: BSAmericanBinaryOption,
            EuropeanBinaryOption: BSEuropeanBinaryOption,
        }[derivative.__class__]
        self.__init__(derivative)
