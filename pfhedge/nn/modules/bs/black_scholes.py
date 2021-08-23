from typing import Callable
from typing import List

from torch import Tensor
from torch.nn import Module

from .american_binary import BSAmericanBinaryOption
from .european import BSEuropeanOption
from .european_binary import BSEuropeanBinaryOption
from .lookback import BSLookbackOption


class BlackScholes(Module):
    """Initialize Black-Scholes formula module from a derivative.

    The ``forward`` method returns the Black-Scholes delta.

    Args:
        derivative (:class:`pfhedge.instruments.Derivative`):
            The derivative to get the Black-Scholes formula.

    Shape:
        - Input : :math:`(N, *, H_{\\mathrm{in}})`, where :math:`*` means any number of
          additional dimensions and :math:`H_{\\mathrm{in}}` is the number of input
          features. See :func:`inputs` for the names of input features.
        - Output : :math:`(N, *, 1)`. All but the last dimension are the same shape
          as the input.

    Examples:

        One can instantiate Black-Scholes module by using a derivative.
        For example, one can instantiate :class:`BSEuropeanOption` using
        a :class:`pfhedge.instruments.EuropeanOption`.
        The ``forward`` method returns delta of the derivative.

        >>> import torch
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> from pfhedge.nn import BlackScholes
        >>>
        >>> derivative = EuropeanOption(BrownianStock(), strike=1.1)
        >>> m = BlackScholes(derivative)
        >>> m
        BSEuropeanOption(strike=1.1000)
        >>> m.inputs()
        ['log_moneyness', 'time_to_maturity', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, 0.1, 0.2],
        ...     [ 0.00, 0.1, 0.2],
        ...     [ 0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[0.4497],
                [0.5126],
                [0.5752]])

        Instantiating :class:`BSLookbackOption` using a
        :class:`pfhedge.instruments.LookbackOption`.

        >>> from pfhedge.instruments import LookbackOption
        >>>
        >>> derivative = LookbackOption(BrownianStock(), strike=1.03)
        >>> m = BlackScholes(derivative)
        >>> m
        BSLookbackOption(strike=1.0300)
        >>> m.inputs()
        ['log_moneyness', 'max_log_moneyness', 'time_to_maturity', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, -0.01, 0.1, 0.2],
        ...     [ 0.00,  0.00, 0.1, 0.2],
        ...     [ 0.01,  0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[...],
                [...],
                [...]])
    """

    inputs: Callable[..., List[str]]  # inputs(self) -> List[str]
    price: Callable[..., Tensor]  # price(self, ...) -> Tensor
    delta: Callable[..., Tensor]  # delta(self, ...) -> Tensor
    gamma: Callable[..., Tensor]  # gamma(self, ...) -> Tensor

    def __new__(cls, derivative):
        return {
            "EuropeanOption": BSEuropeanOption,
            "LookbackOption": BSLookbackOption,
            "AmericanBinaryOption": BSAmericanBinaryOption,
            "EuropeanBinaryOption": BSEuropeanBinaryOption,
        }[derivative.__class__.__name__].from_derivative(derivative)
