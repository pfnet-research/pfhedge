from abc import ABC
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from pfhedge._utils.bisect import bisect
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar

from ..functional import entropic_risk_measure
from ..functional import exp_utility
from ..functional import expected_shortfall
from ..functional import isoelastic_utility
from ..functional import quadratic_cvar


class HedgeLoss(Module, ABC):
    """Base class for hedging criteria."""

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        """Returns the loss of the profit-loss distribution.

        This method should be overridden.

        Args:
            input (torch.Tensor): The distribution of the profit and loss.
            target (torch.Tensor or float, default=0): The target portfolio to replicate.
                Typically, target is the payoff of a derivative.

        Shape:
            - input: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - target: :math:`(N, *)`
            - output: :math:`(*)`

        Returns:
            torch.Tensor
        """

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        """Returns the cash amount which is as preferable as
        the given profit-loss distribution in terms of the loss.

        The output ``cash`` is expected to satisfy the following relation:

        .. code::

            loss(torch.full_like(pl, cash)) = loss(pl)

        By default, the output is computed by binary search.
        If analytic form is known, it is recommended to override this method
        for faster computation.

        Args:
            input (torch.Tensor): The distribution of the profit and loss.
            target (torch.Tensor or float, default=0): The target portfolio to replicate.
                Typically, target is the payoff of a derivative.

        Shape:
            - input: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - target: :math:`(N, *)`
            - output: :math:`(*)`

        Returns:
            torch.Tensor
        """
        pl = input - target
        return bisect(self, self(pl), pl.min(), pl.max())


class EntropicRiskMeasure(HedgeLoss):
    r"""Creates a criterion that measures
    the entropic risk measure.

    The entropic risk measure of the profit-loss distribution
    :math:`\text{pl}` is given by:

    .. math::
        \text{loss}(\text{PL}) = \frac{1}{a}
        \log(- \mathbf{E}[u(\text{PL})]) \,,
        \quad
        u(x) = -\exp(-a x) \,.

    .. seealso::
        - :func:`pfhedge.nn.functional.exp_utility`:
          The corresponding utility function.

    Args:
        a (float, default=1.0): Risk aversion coefficient of
            the exponential utility.
            This parameter should be positive.

    Shape:
        - input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - target: :math:`(N, *)`
        - output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import EntropicRiskMeasure
        ...
        >>> loss = EntropicRiskMeasure()
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(2.0539)
        >>> loss.cash(input)
        tensor(-2.0539)
    """

    def __init__(self, a: float = 1.0) -> None:
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")

        super().__init__()
        self.a = a

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return entropic_risk_measure(input - target, a=self.a)

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -self(input - target)


class EntropicLoss(HedgeLoss):
    r"""Creates a criterion that measures the expected exponential utility.

    The loss of the profit-loss :math:`\text{PL}` is given by:

    .. math::
        \text{loss}(\text{PL}) = -\mathbf{E}[u(\text{PL})] \,,
        \quad
        u(x) = -\exp(-a x) \,.

    .. seealso::
        - :func:`pfhedge.nn.functional.exp_utility`:
          The corresponding utility function.

    Args:
        a (float > 0, default=1.0): Risk aversion coefficient of
            the exponential utility.

    Shape:
        - input: :math:`(N, *)` where
            :math:`*` means any number of additional dimensions.
        - target: :math:`(N, *)`
        - output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import EntropicLoss
        ...
        >>> loss = EntropicLoss()
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(7.7982)
        >>> loss.cash(input)
        tensor(-2.0539)
    """

    def __init__(self, a: float = 1.0) -> None:
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")

        super().__init__()
        self.a = a

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -exp_utility(input - target, a=self.a).mean(0)

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -(-exp_utility(input - target, a=self.a).mean(0)).log() / self.a


class IsoelasticLoss(HedgeLoss):
    r"""Creates a criterion that measures the expected isoelastic utility.

    The loss of the profit-loss :math:`\text{PL}` is given by:

    .. math::
        \text{loss}(\text{PL}) = -\mathbf{E}[u(\text{PL})] \,,
        \quad
        u(x) = \begin{cases}
        x^{1 - a} & a \neq 1 \\
        \log{x} & a = 1
        \end{cases} \,.

    .. seealso::
        - :func:`pfhedge.nn.functional.isoelastic_utility`:
          The corresponding utility function.

    Args:
        a (float): Relative risk aversion coefficient of the isoelastic utility.
            This parameter should satisfy :math:`0 < a \leq 1`.

    Shape:
        - input: :math:`(N, *)` where
            :math:`*` means any number of additional dimensions.
        - target: :math:`(N, *)`
        - output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import IsoelasticLoss
        ...
        >>> loss = IsoelasticLoss(0.5)
        >>> input = torch.arange(1.0, 5.0)
        >>> loss(input)
        tensor(-1.5366)
        >>> loss.cash(input)
        tensor(2.3610)

        >>> loss = IsoelasticLoss(1.0)
        >>> pl = torch.arange(1.0, 5.0)
        >>> loss(input)
        tensor(-0.7945)
        >>> loss.cash(input)
        tensor(2.2134)
    """

    def __init__(self, a: float) -> None:
        if not 0 < a <= 1:
            raise ValueError(
                "Relative risk aversion coefficient should satisfy 0 < a <= 1."
            )

        super().__init__()
        self.a = a

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -isoelastic_utility(input - target, a=self.a).mean(0)


class ExpectedShortfall(HedgeLoss):
    r"""Creates a criterion that measures the expected shortfall.

    .. seealso::
        - :func:`pfhedge.nn.functional.expected_shortfall`

    Args:
        p (float, default=0.1): Quantile level.
            This parameter should satisfy :math:`0 < p \leq 1`.

    Shape:
        - input: :math:`(N, *)` where
            :math:`*` means any number of additional dimensions.
        - target: :math:`(N, *)`
        - output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import ExpectedShortfall
        ...
        >>> loss = ExpectedShortfall(0.5)
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(2.5000)
        >>> loss.cash(input)
        tensor(-2.5000)
    """

    def __init__(self, p: float = 0.1):
        if not 0 < p <= 1:
            raise ValueError("The quantile level should satisfy 0 < p <= 1.")

        super().__init__()
        self.p = p

    def extra_repr(self) -> str:
        return str(self.p)

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return expected_shortfall(input - target, p=self.p, dim=0)

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -self(input - target)


class QuadraticCVaR(HedgeLoss):
    r"""Creates a criterion that measures the QuadraticCVaR.

    .. math::

        \\rho (X) = \\inf_\\omega \\left\\{\\omega + \\lambda || \\min\\{0, X + \\omega\\}||_2\\right\\}.

    for :math:`\lambda\geq1`.

    References:
        - Buehler, Hans, Statistical Hedging (March 1, 2019). Available at SSRN: http://dx.doi.org/10.2139/ssrn.2913250

    .. seealso::
        - :func:`pfhedge.nn.functional.quadratic_cvar`

    Args:
        lam (float, default=10.0): :math:`\lambda`.
            This parameter should satisfy :math:`\lambda \geq 1`.

    Shape:
        - input: :math:`(N, *)` where
            :math:`*` means any number of additional dimensions.
        - target: :math:`(N, *)`
        - output: :math:`(*)`

    Examples:
        >>> from pfhedge.nn import QuadraticCVaR
        ...
        >>> loss = QuadraticCVaR(2.0)
        >>> input = -torch.arange(10.0)
        >>> loss(input)
        tensor(7.9750)
        >>> loss.cash(input)
        tensor(-7.9750)
    """

    def __init__(self, lam: float = 10.0):
        if not lam >= 1.0:
            raise ValueError("The lam should satisfy lam >= 1.")

        super().__init__()
        self.lam = lam

    def extra_repr(self) -> str:
        return str(self.lam)

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return quadratic_cvar(input - target, lam=self.lam, dim=0)

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -self(input - target)


class OCE(HedgeLoss):
    r"""Creates a criterion that measures the optimized certainty equivalent.

    The certainty equivalent is given by:

    .. math::
        \text{loss}(X, w) = w - \mathrm{E}[u(X + w)]

    Minimization of loss gives the optimized certainty equivalent.

    .. math::
        \rho_u(X) = \inf_w \text{loss}(X, w)

    Args:
        utility (callable): Utility function.

    Attributes:
        w (torch.nn.Parameter): Represents wealth.

    Examples:
        >>> from pfhedge.nn.modules.loss import OCE
        ...
        >>> _ = torch.manual_seed(42)
        >>> m = OCE(lambda x: 1 - (-x).exp())
        >>> pl = torch.randn(10)
        >>> m(pl)
        tensor(0.0855, grad_fn=<SubBackward0>)
        >>> m.cash(pl)
        tensor(-0.0821)
    """

    def __init__(self, utility: Callable[[Tensor], Tensor]) -> None:
        super().__init__()

        self.utility = utility
        self.w = Parameter(torch.tensor(0.0))

    def extra_repr(self) -> str:
        w = float(self.w.item())
        return self.utility.__name__ + ", w=" + _format_float(w)

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return self.w - self.utility(input - target + self.w).mean(0)
