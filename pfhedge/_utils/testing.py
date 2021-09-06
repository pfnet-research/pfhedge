from typing import Callable

import torch
from torch import Tensor
from torch.testing import assert_close


def assert_monotone(
    fn: Callable[[Tensor], Tensor],
    x1: Tensor,
    x2: Tensor,
    increasing: bool = False,
    allow_equal: bool = False,
) -> None:
    """Assert ``fn`` is monotone function for :math:`x_1 > x_2`:

    .. math::
        f(x_1) > f(x_2)

    Args:
        fn (callable[[torch.Tensor], torch.Tensor]): Function to test.
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.
        increasing (bool, default=True): Whether to test increasing monotonicity.
            Only False is supported.
        allow_equal (bool, default=False): Whether to allow :math:`f(x_1) = f(x_2)`.
            Only False is supported.

    Returns:
        None

    Raises:
        AssertionError: If fn is not monotone.
    """
    assert not increasing, "not supported"
    assert not allow_equal, "not supported"
    assert (x1 > x2).all(), "x1 > x2 must be satisfied"

    assert (fn(x1) < fn(x2)).all()


def assert_convex(
    fn: Callable[[Tensor], Tensor], x1: Tensor, x2: Tensor, alpha: float
) -> None:
    """Assert convexity.

    .. math::
        f(\\alpha * x_1 + (1 - \\alpha) * x_2) \\leq
        \\alpha * f(x_1) + (1 - \\alpha) * f(x_2)

    Args:
        fn (callable[[torch.Tensor], torch.Tensor]): Function to test.
            It should return a tensor with a single element.
        x1 (torch.Tensor): The first tensor.
        x2 (torch.Tensor): The second tensor.
        alpha (float): The parameter alpha.
    """
    y = fn(alpha * x1 + (1 - alpha) * x2)
    y1 = fn(x1)
    y2 = fn(x2)
    assert y <= alpha * y1 + (1 - alpha) * y2


def assert_cash_invariant(
    fn: Callable[[Tensor], Tensor], x: Tensor, c: float, **kwargs
) -> None:
    """Assert cash invariance.

    .. math::
        f(x + c) = f(x) - c

    Args:
        fn (callable): Function to test cash invariance.
        x (torch.Tensor): The input tensor.
        c (float): The parameter c.
    """
    assert_close(fn(x + c), fn(x) - c, **kwargs)


def assert_cash_equivalent(
    fn: Callable[[Tensor], Tensor], x: Tensor, c: float, **kwargs
) -> None:
    """Assert ``c`` is the cash equivalent of ``x``.
    ``fn(x) = fn(torch.full_like(x, c))``

    Args:
        fn (callable): Function to test cash equivalent.
        x (torch.Tensor): The input tensor.
        c (float): The parameter c.
        **kwargs: Keyword arguments to pass to ``assert_close``.
    """
    assert_close(fn(x), fn(torch.full_like(x, c)), **kwargs)
