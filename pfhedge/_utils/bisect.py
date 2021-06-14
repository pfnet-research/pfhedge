import torch
from torch import Tensor


def bisect(
    function, target: Tensor, lower, upper, precision: float = 1e-6, abort: int = 100000
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
        abort (int, default 100000): If the number of iterations exceeds this
            value, abort computation and raise RuntimeError.

    Returns:
        torch.Tensor

    Raises:
        RuntimeError: If the number of iteration exceeds `abort`.

    Examples:

        >>> target = torch.tensor([-1.0, 0.0, 1.0])
        >>> function = torch.log
        >>> output = bisect(function, target, 0.01, 10.0)
        >>> output
        tensor([0.3679, 1.0000, 2.7183])
        >>> torch.allclose(function(output), target, atol=1e-6)
        True

        Monotone decreasing function:

        >>> function = lambda x: -torch.log(x)
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
        # If decreasing function
        mf = lambda x: -function(x)
        return bisect(mf, -target, lower, upper, precision=precision, abort=abort)

    lower = torch.full_like(target, lower)
    upper = torch.full_like(target, upper)

    n_iter = 0
    while torch.max(upper - lower) > precision:
        n_iter += 1
        if n_iter > abort:
            raise RuntimeError(f"Aborting since iteration exceeds abort={abort}.")

        m = (lower + upper) / 2
        output = function(m)
        lower = torch.where(output < target, m, lower)
        upper = torch.where(output >= target, m, upper)

    return upper
