import torch

from ..bs.bs import BlackScholes
from .clamp import Clamp


class WhalleyWilmott(torch.nn.Module):
    """Initialize Whalley-Wilmott's hedging strategy of a derivative.

    The `forward` method returns the next hedge ratio.

    This is the optimal hedging strategy for asymptotically small transaction cost.

    Args:
        derivative (:class:`pfhedge.instruments.Derivative`): Derivative to hedge.
        a (float, default=1.0): Risk aversion parameter in exponential utility.

    Shape:
        - Input: :math:`(N, *, H_{\\text{in}})`.  Here, :math:`*` means any number of
          additional dimensions and `H_in` is the number of input features.
          See `features()` for input features.
        - Output: :math:`(N, *, 1)`. The hedge ratio at the next time step.

    Examples:

        An example for :class:`pfhedge.instruments.EuropeanOption`.

        >>> from pfhedge.nn import WhalleyWilmott
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> deriv = EuropeanOption(BrownianStock(cost=1e-5))
        >>> m = WhalleyWilmott(deriv)
        >>> m.features()
        ['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge']
        >>> input = torch.tensor([
        ...     [-0.05, 0.1, 0.2, 0.5],
        ...     [-0.01, 0.1, 0.2, 0.5],
        ...     [ 0.00, 0.1, 0.2, 0.5],
        ...     [ 0.01, 0.1, 0.2, 0.5],
        ...     [ 0.05, 0.1, 0.2, 0.5]])
        >>> m(input)
        tensor([[0.2946],
                [0.5000],
                [0.5000],
                [0.5000],
                [0.7284]])

        An example for :class:`pfhedge.instruments.EuropeanOption` without cost.

        >>> deriv = EuropeanOption(BrownianStock())
        >>> m = WhalleyWilmott(deriv)
        >>> m.features()
        ['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge']
        >>> input = torch.tensor([
        ...     [-0.05, 0.1, 0.2, 0.5],
        ...     [-0.01, 0.1, 0.2, 0.5],
        ...     [ 0.00, 0.1, 0.2, 0.5],
        ...     [ 0.01, 0.1, 0.2, 0.5],
        ...     [ 0.05, 0.1, 0.2, 0.5]])
        >>> m(input)
        tensor([[0.2239],
                [0.4497],
                [0.5126],
                [0.5752],
                [0.7945]])

    References:
        - Whalley, A.E. and Wilmott, P., An asymptotic analysis of an optimal hedging model
          for option pricing with transaction costs. Mathematical Finance, 1997, 7, 307–324.
    """

    def __init__(self, derivative, a: float = 1.0):
        super().__init__()
        self.derivative = derivative
        self.a = a

        self.bs = BlackScholes(derivative)
        self.clamp = Clamp()

    def features(self):
        return self.bs.features() + ["prev_hedge"]

    def extra_repr(self):
        return f"a={self.a}" if self.a != 1 else ""

    def forward(self, input) -> torch.Tensor:
        prev_hedge = input[..., [-1]]

        delta = self.bs(input[..., :-1])
        width = self.width(input[..., :-1])

        return self.clamp(prev_hedge, min=delta - width, max=delta + width)

    def width(self, input) -> torch.Tensor:
        """Returns half-width of the no-transaction band.

        Args:
            input : Tensor

        Shape:
            - Input: :math:`(N, *, H_{\\text{in}} - 1)`
            - Output: :math:`(N, *, 1)`

        Returns:
            torch.Tensor
        """
        cost = self.derivative.underlier.cost

        price = self.derivative.strike * torch.exp(input[..., [0]])
        gamma = self.bs.gamma(*(input[..., [i]] for i in range(input.size()[-1])))
        width = (cost * (3 / 2) * (gamma ** 2) * price / self.a) ** (1 / 3)

        return width
