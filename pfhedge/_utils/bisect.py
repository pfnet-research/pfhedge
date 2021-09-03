from typing import Callable
from typing import Union

import torch
from torch import Tensor


def bisect(
    fn: Callable[[Tensor], Tensor],
    target: Tensor,
    lower: Union[float, Tensor],
    upper: Union[float, Tensor],
    precision: float = 1e-6,
    max_iter: int = 100000,
) -> Tensor:
    """Perform binary search over a tensor.

    The output tensor approximately satisfies the following relation:

    .. code-block::

        fn(output) = target

    Args:
        fn (callable[[Tensor], Tensor]): A monotone function.
        target (Tensor): Target of function values.
        lower (Tensor or float): Lower bound of binary search.
        upper (Tensor or float): Upper bound of binary search.
        precision (float, default=1e-6): Precision of output.
        max_iter (int, default 100000): If the number of iterations exceeds this
            value, abort computation and raise RuntimeError.

    Returns:
        torch.Tensor

    Raises:
        RuntimeError: If the number of iteration exceeds ``max_iter``.

    Examples:

        >>> target = torch.tensor([-1.0, 0.0, 1.0])
        >>> fn = torch.log
        >>> output = bisect(fn, target, 0.01, 10.0)
        >>> output
        tensor([0.3679, 1.0000, 2.7183])
        >>> torch.allclose(fn(output), target, atol=1e-6)
        True

        Monotone decreasing function:

        >>> fn = lambda input: -torch.log(input)
        >>> output = bisect(fn, target, 0.01, 10.0)
        >>> output
        tensor([2.7183, 1.0000, 0.3679])
        >>> torch.allclose(fn(output), target, atol=1e-6)
        True
    """
    lower, upper = map(torch.as_tensor, (lower, upper))

    if not (lower < upper).all():
        raise ValueError("condition lower < upper should be satisfied.")

    if (fn(lower) > fn(upper)).all():
        # If fn is a decreasing function
        mf = lambda input: -fn(input)
        return bisect(mf, -target, lower, upper, precision=precision, max_iter=max_iter)

    n_iter = 0
    while torch.max(upper - lower) > precision:
        n_iter += 1
        if n_iter > max_iter:
            raise RuntimeError(f"Aborting since iteration exceeds max_iter={max_iter}.")

        m = (lower + upper) / 2
        output = fn(m)
        lower = lower.where(output >= target, m)
        upper = upper.where(output < target, m)

    return upper


def find_implied_volatility(
    pricer: Callable,
    price: Tensor,
    lower: float = 0.001,
    upper: float = 1.000,
    precision: float = 1e-6,
    max_iter: int = 100,
    **params,
) -> Tensor:
    """Find implied volatility by binary search.

    Args:
        pricer (callable): Pricing formula of a derivative.
        price (Tensor): The price of the derivative.
        lower (float, default=0.001): Lower bound of binary search.
        upper (float, default=1.000): Upper bound of binary search.
        precision (float, default=1e-6): Computational precision of the implied
            volatility.
        max_iter (int, default 100): If the number of iterations exceeds this
            value, abort computation and raise RuntimeError.

    Returns:
        torch.Tensor
    """
    fn = lambda volatility: pricer(volatility=volatility, **params)
    return bisect(fn, price, lower, upper, precision=precision, max_iter=max_iter)
