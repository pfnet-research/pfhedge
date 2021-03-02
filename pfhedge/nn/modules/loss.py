from abc import ABC

import torch
from torch.nn import Parameter

from ..._utils.bisect import bisect
from ..functional import exp_utility
from ..functional import expected_shortfall
from ..functional import isoelastic_utility


class HedgeLoss(torch.nn.Module, ABC):
    """
    Base class for hedging criteria.

    Shape
    -----
    Input : (N, *)
        Profit and loss distribution.
        Here, `*` means any number of additional dimensions.
    Output : (*)
        Loss.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Returns the loss of the profit-loss distribution.
        This method should be overridden.

        Parameters
        ----------
        - input : Tensor, shape (N, *)
            The distribution of the profit and loss.
            Here, `*` means any number of additional dimensions.

        Returns
        -------
        loss : Tensor, shape (*)
        """

    def cash(self, input: torch.Tensor) -> torch.Tensor:
        """
        Returns the cash amount which is as preferable as
        the given profit-loss distribution in terms of the loss.

        The output `cash` is expected to satisfy the following relation:

            loss(torch.full_like(pnl, cash)) = loss(pnl)

        By default, the output is computed by binary search.
        If analytic form is known, it is recommended to override this method
        for faster computation.

        Parameters
        ----------
        - input : Tensor, shape (N, *)
            The distribution of the profit and loss.
            Here, `*` means any number of additional dimensions.

        Returns
        -------
        cash : Tensor, shape (*)
        """
        return bisect(self, self(input), torch.min(input), torch.max(input))


class EntropicRiskMeasure(HedgeLoss):
    """
    Creates a loss given by the entropic risk measure.

    The entropic risk measure of the profit-loss distribution `pnl` is given by

        loss(pnl) = log(-E[u(pnl)]) / a
        u(x) = -exp(-a * x)

    Parameters
    ----------
    - a : float > 0, default 1.0
        Risk aversion coefficient of the exponential utility.

    Shape
    -----
    Input : (N, *)
        The distribution of the profit and loss.
        Here, `*` means any number of additional dimensions.
    Output : (*)
        Loss.

    Shape
    -----
    >>> loss = EntropicRiskMeasure()
    >>> pnl = -torch.arange(4.0)
    >>> loss(pnl)
    tensor(2.0539)
    >>> loss.cash(pnl)
    tensor(-2.0539)
    """

    def __init__(self, a=1.0):
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")

        super().__init__()
        self.a = a

    def extra_repr(self):
        return f"a={self.a}" if self.a != 1 else ""

    def forward(self, input):
        return torch.log(-torch.mean(exp_utility(input, a=self.a), dim=0)) / self.a

    def cash(self, input):
        """
        Returns the cash amount which is as preferable as
        the given profit-loss distribution in terms of the loss.

        Parameters
        ----------
        - input : Tensor, shape (N, *)
            The distribution of the profit and loss.
            Here, `*` means any number of additional dimensions.

        Returns
        -------
        cash : Tensor, shape (*)
        """
        return -self(input)


class EntropicLoss(HedgeLoss):
    """
    Creates a loss given by the negative of expected exponential utility.

    The loss of the profit-loss `pnl` is given by

        loss(pnl) = -E[u(pnl)]
        u(x) = -exp(-a * x)

    Parameters
    ----------
    - a : float > 0, default 1.0
        Risk aversion coefficient of the exponential utility.

    Shape
    -----
    Input : (N, *)
        The distribution of the profit and loss.
        Here, `*` means any number of additional dimensions.
    Output : (*)
        Loss.

    Examples
    --------
    >>> loss = EntropicLoss()
    >>> pnl = -torch.arange(4.0)
    >>> loss(pnl)
    tensor(7.7982)
    >>> loss.cash(pnl)
    tensor(-2.0539)
    """

    def __init__(self, a=1.0):
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")

        super().__init__()
        self.a = a

    def extra_repr(self):
        return f"a={self.a}" if self.a != 1 else ""

    def forward(self, input):
        return -torch.mean(exp_utility(input, a=self.a), dim=0)

    def cash(self, input):
        """
        Returns the cash amount which is as preferable as
        the given profit-loss distribution in terms of the loss.

        Parameters
        ----------
        - input : Tensor, shape (N, *)
            The distribution of the profit and loss.
            Here, `*` means any number of additional dimensions.

        Returns
        -------
        cash : Tensor, shape (*)
        """
        return -torch.log(-torch.mean(exp_utility(input, a=self.a), dim=0)) / self.a


class IsoelasticLoss(HedgeLoss):
    """
    Creates a loss function that measures the isoelastic utility.

    The loss of the profit-loss distribution `pnl` is given by

        loss(pnl) = -E[u(pnl)]

        u(x) = x ** (1 - a) if a != 1
               log(x) if a == 1

    Parameters
    ----------
    - a : float
        Relative risk aversion coefficient of the isoelastic utility.
        This parameter should satisfy `0 < a <= 1`.

    Shape
    -----
    Input : (N, *)
        The distribution of the profit and loss.
        Here, `*` means any number of additional dimensions.
    Output : (*)
        Loss.

    Examples
    --------
    >>> loss = IsoelasticLoss(0.5)
    >>> pnl = torch.arange(1.0, 5.0)
    >>> loss(pnl)
    tensor(-1.5366)
    >>> loss.cash(pnl)
    tensor(2.3610)

    >>> loss = IsoelasticLoss(1.0)
    >>> pnl = torch.arange(1.0, 5.0)
    >>> loss(pnl)
    tensor(-0.7945)
    >>> loss.cash(pnl)
    tensor(2.2134)
    """

    def __init__(self, a):
        if not 0 < a <= 1:
            raise ValueError(
                "Relative risk aversion coefficient should satisfy 0 < a <= 1."
            )

        super().__init__()
        self.a = a

    def extra_repr(self):
        return f"a={self.a}"

    def forward(self, input):
        return -torch.mean(isoelastic_utility(input, a=self.a), dim=0)


class ExpectedShortfall(HedgeLoss):
    """
    Creates a criterion that measures the expected shortfall.

    Parameters
    ----------
    - p : float in (0.0, 1.0], default 0.1
        Quantile level.
        This parameter should satisfy 0 < p <= 1.

    Shape
    -----
    Input : (N, *)
        The distribution of the profit and loss.
        Here, `*` means any number of additional dimensions.
    Output : (*)
        Loss.

    Examples
    --------
    >>> loss = ExpectedShortfall(0.5)
    >>> pnl = -torch.arange(4.0)
    >>> loss(pnl)
    tensor(2.5000)
    >>> loss.cash(pnl)
    tensor(-2.5000)
    """

    def __init__(self, p=0.1):
        if not 0 < p <= 1:
            raise ValueError("The quantile level should satisfy 0 < p <= 1.")

        super().__init__()
        self.p = p

    def extra_repr(self):
        return str(self.p)

    def forward(self, input):
        return expected_shortfall(input, p=self.p, dim=0)

    def cash(self, input):
        return -self(input)


class OCE(HedgeLoss):
    """
    Creates a criterion that measures the optimized certainty equivalent.

    The certainty equivalent is given by:

        loss(X, w) = w - E[u(X + w)]

    Minimization of loss gives the optimized certainty equivalent.

        rho_u(X) = inf_w loss(X, w)

    Parameters
    ----------
    - utility : torch.autograd.Function
        Utility function.

    Attributes
    ----------
    - w : torch.nn.Parameter
        Wealth.

    Examples
    --------
    >>> _ = torch.manual_seed(42)
    >>> m = OCE(lambda x: 1 - torch.exp(-x))
    >>> pnl = torch.randn(10)
    >>> m(pnl)
    tensor(0.0855, grad_fn=<SubBackward0>)
    >>> m.cash(pnl)
    tensor(-0.0821)
    """

    def __init__(self, utility):
        super().__init__()

        self.utility = utility
        self.w = Parameter(torch.tensor(0.0))

    def extra_repr(self):
        return self.utility.__name__ + f", w={self.w}"

    def forward(self, input):
        return self.w - torch.mean(self.utility(input + self.w), dim=0)
