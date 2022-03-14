from torch import Tensor
from torch.nn import Module

from pfhedge._utils.typing import TensorOrScalar
from pfhedge.nn.functional import svi_variance


class SVIVariance(Module):
    r"""Returns total variance in the SVI model.

    The total variance for log strike :math:`k = \log(K / S)`,
    where :math:`K` and :math:`S` are strike and spot, reads:

    .. math::
        w = a + b \left[ \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right] .

    References:
        - Jim Gatheral and Antoine Jacquier,
          Arbitrage-free SVI volatility surfaces.
          [arXiv:`1204.0646 <https://arxiv.org/abs/1204.0646>`_ [q-fin.PR]]

    Args:
        a (torch.Tensor or float): The parameter :math:`a`.
        b (torch.Tensor or float): The parameter :math:`b`.
        rho (torch.Tensor or float): The parameter :math:`\rho`.
        m (torch.Tensor or float): The parameter :math:`m`.
        sigma (torch.Tensor or float): The parameter :math:`\sigma`.

    Examples:
        >>> import torch
        >>>
        >>> a, b, rho, m, sigma = 0.03, 0.10, 0.10, 0.00, 0.10
        >>> module = SVIVariance(a, b, rho, m, sigma)
        >>> input = torch.tensor([-0.10, -0.01, 0.00, 0.01, 0.10])
        >>> module(input)
        tensor([0.0431, 0.0399, 0.0400, 0.0401, 0.0451])
    """

    def __init__(
        self,
        a: TensorOrScalar,
        b: TensorOrScalar,
        rho: TensorOrScalar,
        m: TensorOrScalar,
        sigma: TensorOrScalar,
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma

    def forward(self, input: Tensor) -> Tensor:
        return svi_variance(
            input, a=self.a, b=self.b, rho=self.rho, m=self.m, sigma=self.sigma
        )

    def extra_repr(self) -> str:
        params = (
            f"a={self.a}",
            f"b={self.b}",
            f"rho={self.rho}",
            f"m={self.m}",
            f"sigma={self.sigma}",
        )
        return ", ".join(params)
