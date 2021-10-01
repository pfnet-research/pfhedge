from typing import List

from torch import Tensor
from torch.nn import Module

from pfhedge._utils.str import _format_float

from .bs.black_scholes import BlackScholes
from .clamp import Clamp


class WhalleyWilmott(Module):
    r"""Initialize Whalley-Wilmott's hedging strategy of a derivative.

    The ``forward`` method returns the next hedge ratio.

    This is the no-transaction band strategy
    that is optimal under the premises of
    asymptotically small transaction cost, European option, and exponential utility.
    The half-width of the no-transaction band is given by

    .. math::

        w = \left( \frac{3 c \Gamma^2 S}{2 a} \right)^{1 / 3} \,,

    where :math:`c` is the transaction cost rate,
    :math:`\Gamma` is the gamma of the derivative,
    :math:`S` is the spot price of the underlying instrument, and
    :math:`a` is the risk-aversion coefficient of the exponential utility.

    Note:
        A backward computation for this module generates ``nan``
        if the :math:`\Gamma` of the derivative is too small.
        This is because the output is proportional to :math:`\Gamma^{2 / 3}`
        of which gradient diverges for :math:`\Gamma \to 0`.
        A ``dtype`` with higher precision may alleviate this problem.

    References:
        - Davis, M.H., Panas, V.G. and Zariphopoulou, T., 1993.
          European option pricing with transaction costs.
          SIAM Journal on Control and Optimization, 31(2), pp.470-493.
        - Whalley, A.E. and Wilmott, P., An asymptotic analysis of an optimal hedging
          model for option pricing with transaction costs. Mathematical Finance,
          1997, 7, 307–324.

    Args:
        derivative (:class:`pfhedge.instruments.Derivative`): Derivative to hedge.
        a (float, default=1.0): Risk aversion parameter in exponential utility.

    Shape:
        - Input: :math:`(N, *, H_{\text{in}})` where
          :math:`*` means any number of additional dimensions and
          :math:`H_{\text{in}}` is the number of input features.
          See :meth:`inputs()` for the names of input features.
        - Output: :math:`(N, *, 1)`.

    Examples:

        An example for :class:`pfhedge.instruments.EuropeanOption`.

        >>> import torch
        >>> from pfhedge.nn import WhalleyWilmott
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> derivative = EuropeanOption(BrownianStock(cost=1e-5))
        >>>
        >>> m = WhalleyWilmott(derivative)
        >>> m.inputs()
        ['log_moneyness', 'time_to_maturity', 'volatility', 'prev_hedge']
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

        >>> derivative = EuropeanOption(BrownianStock())
        >>> m = WhalleyWilmott(derivative)
        >>> m.inputs()
        ['log_moneyness', 'time_to_maturity', 'volatility', 'prev_hedge']
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
    """

    def __init__(self, derivative, a: float = 1.0) -> None:
        super().__init__()
        self.derivative = derivative
        self.a = a

        self.bs = BlackScholes(derivative)
        self.clamp = Clamp()

    def inputs(self) -> List[str]:
        """Returns the names of input features.

        Returns:
            list[str]
        """
        return self.bs.inputs() + ["prev_hedge"]

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor) -> Tensor:
        prev_hedge = input[..., [-1]]

        delta = self.bs(input[..., :-1])
        width = self.width(input[..., :-1])
        min = delta - width
        max = delta + width

        return self.clamp(prev_hedge, min=min, max=max)

    def width(self, input: Tensor) -> Tensor:
        r"""Returns half-width of the no-transaction band.

        Args:
            input (Tensor): The input tensor.

        Shape:
            - Input: :math:`(N, *, H_{\text{in}} - 1)` where
              :math:`*` means any number of additional dimensions and
              :math:`H_{\text{in}}` is the number of input features.
              See :meth:`inputs()` for the names of input features.
            - Output: :math:`(N, *, 1)`

        Returns:
            torch.Tensor
        """
        cost = self.derivative.underlier.cost

        spot = self.derivative.strike * input[..., [0]].exp()
        gamma = self.bs.gamma(*(input[..., [i]] for i in range(input.size(-1))))
        width = (cost * (3 / 2) * gamma.square() * spot / self.a).pow(1 / 3)

        return width
