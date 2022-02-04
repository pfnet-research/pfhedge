from torch.nn import Module

from pfhedge._utils.typing import TensorOrScalar
from pfhedge.nn.functional import svi_sigma


class SVISigma(Module):
    r"""Returns volatility in the SVI model.

    The variance for log moneyness :math:`x` reads:

    .. math::
        \sigma^2 = a + b \left[ \rho (x - m) + \sqrt{(x - m)^2 + s^2} \right]

    References:
        - Jim Gatheral and Antoine Jacquier,
          Arbitrage-free SVI volatility surfaces.
          [arXiv:`1204.0646 <https://arxiv.org/abs/1204.0646>`_ [q-fin.PR]]

    Args:
        a (torch.Tensor or float): The parameter :math:`a`.
        b (torch.Tensor or float): The parameter :math:`b`.
        rho (torch.Tensor or float): The parameter :math:`\rho`.
        m (torch.Tensor or float): The parameter :math:`m`.
        s (torch.Tensor or float): The parameter :math:`s`.

    Examples:
        >>> import torch
        >>> a, b, rho, m, s = 0.04, 0.20, -0.10, 0.00, 0.00
        >>> module = SVISigma(a, b, rho, m, s)
        >>> input = torch.tensor([-0.10, -0.01, 0.00, 0.01, 0.10])
        >>> module(input)
        tensor([0.2098, 0.2005, 0.2000, 0.1995, 0.2000])
    """

    def __init__(
        self,
        a: TensorOrScalar,
        b: TensorOrScalar,
        rho: TensorOrScalar,
        m: TensorOrScalar,
        s: TensorOrScalar,
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.s = s

    def forward(self, input) -> Tensor:
        return svi_sigma(input, a=self.a, b=self.b, rho=self.rho, m=self.m, s=self.s)
