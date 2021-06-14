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
    """Initialize Black-Scholes formula module from a derivative.

    The `forward` method returns the Black-Scholes delta.

    Args:
        derivative (:class:`pfhedge.instruments.Derivative`):
            The derivative to get the Black-Scholes formula.

    Shape:
        - Input : :math:`(N, *, H_{\\mathrm{in}})`, where :math:`*` means any number of
          additional dimensions and :math:`H_{\\mathrm{in}}` is the number of input
          features. See `features()` for input features.
        - Output : :math:`(N, *, 1)`. All but the last dimension are the same shape
          as the input.

    Examples:

        One can instantiate Black-Scholes module by using a derivative.
        For example, one can instantiate :class:`BSEuropeanOption` using
        a :class:`pfhedge.instruments.EuropeanOption`.
        The `forward` method returns delta of the derivative.

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> from pfhedge.nn import BlackScholes
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

        Instantiating :class:`BSLookbackOption` using a
        :class:`pfhedge.instruments.LookbackOption`.

        >>> from pfhedge.instruments import LookbackOption
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
