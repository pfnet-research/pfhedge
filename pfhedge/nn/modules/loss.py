from abc import ABC
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

from pfhedge._utils.bisect import bisect
from pfhedge._utils.str import _format_float

from ..functional import exp_utility
from ..functional import expected_shortfall
from ..functional import isoelastic_utility


class HedgeLoss(Module, ABC):
    """Base class for hedging criteria."""

    def forward(self, input: Tensor) -> Tensor:
        """Returns the loss of the profit-loss distribution.

        This method should be overridden.

        Args:
            input (torch.Tensor): The distribution of the profit and loss.

        Shape:
            - Input: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - Output: :math:`(*)`

        Returns:
            torch.Tensor
        """

    def cash(self, input: Tensor) -> Tensor:
        """Returns the cash amount which is as preferable as
        the given profit-loss distribution in terms of the loss.

        The output ``cash`` is expected to satisfy the following relation:

        .. code::

            loss(torch.full_like(pnl, cash)) = loss(pnl)

        By default, the output is computed by binary search.
        If analytic form is known, it is recommended to override this method
        for faster computation.

        Args:
            input (torch.Tensor): The distribution of the profit and loss.

        Shape:
            - Input: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - Output: :math:`(*)`

        Returns:
            torch.Tensor
        """
        return bisect(self, self(input), input.min(), input.max())


class EntropicRiskMeasure(HedgeLoss):
    r"""Creates a loss given by the entropic risk measure.

    The entropic risk measure of the profit-loss distribution
    :math:`\text{pnl}` is given by:

    .. math::

        \text{loss}(\text{pnl}) = \frac{1}{a}
        \log(- \mathbf{E}[u(\text{pnl})]) \,,
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
        - Input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples:

        >>> from pfhedge.nn import EntropicRiskMeasure
        >>>
        >>> loss = EntropicRiskMeasure()
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(2.0539)
        >>> loss.cash(input)
        tensor(-2.0539)
    """

    def __init__(self, a: float = 1.0):
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")

        super().__init__()
        self.a = a

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor) -> Tensor:
        return (-exp_utility(input, a=self.a).mean(0)).log() / self.a

    def cash(self, input: Tensor) -> Tensor:
        return -self(input)


class EntropicLoss(HedgeLoss):
    r"""Creates a loss given by the negative of expected exponential utility.

    The loss of the profit-loss :math:`\text{pnl}` is given by:

    .. math::

        \text{loss}(\text{pnl}) = -\mathbf{E}[u(\text{pnl})] \,,
        \quad
        u(x) = -\exp(-a x) \,.

    .. seealso::
        - :func:`pfhedge.nn.functional.exp_utility`:
          The corresponding utility function.

    Args:
        a (float > 0, default=1.0): Risk aversion coefficient of
            the exponential utility.

    Shape:
        - Input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples:

        >>> from pfhedge.nn import EntropicLoss
        >>>
        >>> loss = EntropicLoss()
        >>> input = -torch.arange(4.0)
        >>> loss(input)
        tensor(7.7982)
        >>> loss.cash(input)
        tensor(-2.0539)
    """

    def __init__(self, a: float = 1.0):
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")

        super().__init__()
        self.a = a

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor) -> Tensor:
        return -exp_utility(input, a=self.a).mean(0)

    def cash(self, input: Tensor) -> Tensor:
        return -(-exp_utility(input, a=self.a).mean(0)).log() / self.a


class IsoelasticLoss(HedgeLoss):
    r"""Creates a loss function that measures the isoelastic utility.

    The loss of the profit-loss :math:`\text{pnl}` is given by:

    .. math::

        \text{loss}(\text{pnl}) = -\mathbf{E}[u(\text{pnl})] \,,
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
        - Input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples:

        >>> from pfhedge.nn import IsoelasticLoss
        >>>
        >>> loss = IsoelasticLoss(0.5)
        >>> input = torch.arange(1.0, 5.0)
        >>> loss(input)
        tensor(-1.5366)
        >>> loss.cash(input)
        tensor(2.3610)

        >>> loss = IsoelasticLoss(1.0)
        >>> pnl = torch.arange(1.0, 5.0)
        >>> loss(input)
        tensor(-0.7945)
        >>> loss.cash(input)
        tensor(2.2134)
    """

    def __init__(self, a: float):
        if not 0 < a <= 1:
            raise ValueError(
                "Relative risk aversion coefficient should satisfy 0 < a <= 1."
            )

        super().__init__()
        self.a = a

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor) -> Tensor:
        return -isoelastic_utility(input, a=self.a).mean(0)


class ExpectedShortfall(HedgeLoss):
    r"""Creates a criterion that measures the expected shortfall.

    .. seealso::
        - :func:`pfhedge.nn.functional.expected_shortfall`

    Args:
        p (float, default=0.1): Quantile level.
            This parameter should satisfy :math:`0 < p \leq 1`.

    Shape:
        - Input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - Output: :math:`(*)`

    Examples:

        >>> from pfhedge.nn import ExpectedShortfall
        >>>
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

    def forward(self, input: Tensor) -> Tensor:
        return expected_shortfall(input, p=self.p, dim=0)

    def cash(self, input: Tensor) -> Tensor:
        return -self(input)


class OCE(HedgeLoss):
    """Creates a criterion that measures the optimized certainty equivalent.

    The certainty equivalent is given by:

    .. math::

        \\text{loss}(X, w) = w - \\mathrm{E}[u(X + w)]

    Minimization of loss gives the optimized certainty equivalent.

    .. math::

        \\rho_u(X) = \\inf_w \\text{loss}(X, w)

    Args:
        utility (callable): Utility function.

    Attributes:
        w (torch.nn.Parameter): Represents wealth.

    Examples:

        >>> from pfhedge.nn.modules.loss import OCE
        >>>
        >>> _ = torch.manual_seed(42)
        >>> m = OCE(lambda x: 1 - (-x).exp())
        >>> pnl = torch.randn(10)
        >>> m(pnl)
        tensor(0.0855, grad_fn=<SubBackward0>)
        >>> m.cash(pnl)
        tensor(-0.0821)
    """

    def __init__(self, utility: Callable[[Tensor], Tensor]) -> None:
        super().__init__()

        self.utility = utility
        self.w = Parameter(torch.tensor(0.0))

    def extra_repr(self) -> str:
        w = float(self.w.item())
        return self.utility.__name__ + ", w=" + _format_float(w)

    def forward(self, input: Tensor) -> Tensor:
        return self.w - self.utility(input + self.w).mean(0)
