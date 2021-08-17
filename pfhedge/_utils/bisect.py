from typing import Callable
from typing import Union

import torch
from torch import Tensor


def bisect(
    function: Callable[[Tensor], Tensor],
    target: Tensor,
    lower: Union[float, Tensor],
    upper: Union[float, Tensor],
    precision: float = 1e-6,
    max_iter: int = 100000,
) -> Tensor:
    """Perform binary search over a tensor.

    The output tensor approximately satisfies the following relation:

    .. code-block ::

        function(output) = target

    Args:
        function (callable[[Tensor], Tensor]): Monotone increasing or decreasing
            function.
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
        >>> function = torch.log
        >>> output = bisect(function, target, 0.01, 10.0)
        >>> output
        tensor([0.3679, 1.0000, 2.7183])
        >>> torch.allclose(function(output), target, atol=1e-6)
        True

        Monotone decreasing function:

        >>> function = lambda input: -torch.log(input)
        >>> output = bisect(function, target, 0.01, 10.0)
        >>> output
        tensor([2.7183, 1.0000, 0.3679])
        >>> torch.allclose(function(output), target, atol=1e-6)
        True
    """
    lower, upper = map(torch.as_tensor, (lower, upper))

    if not (lower < upper).all():
        raise ValueError("condition lower < upper should be satisfied.")

    if (function(lower) > function(upper)).all():
        # If function is a decreasing function
        mf = lambda input: -function(input)
        return bisect(mf, -target, lower, upper, precision=precision, max_iter=max_iter)

    n_iter = 0
    while torch.max(upper - lower) > precision:
        n_iter += 1
        if n_iter > max_iter:
            raise RuntimeError(f"Aborting since iteration exceeds max_iter={max_iter}.")

        m = (lower + upper) / 2
        output = function(m)
        lower = lower.where(output >= target, m)
        upper = upper.where(output < target, m)

    return upper
