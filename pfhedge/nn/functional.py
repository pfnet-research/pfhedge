from math import ceil

import torch
import torch.nn.functional as fn
from torch import Tensor


def european_payoff(input: Tensor, call: bool = True, strike: float = 1.0) -> Tensor:
    """Returns the payoff of a European option.

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(T, *)`, where, :math:`T` stands for the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return fn.relu(input[-1] - strike)
    else:
        return fn.relu(strike - input[-1])


def lookback_payoff(input: Tensor, call: bool = True, strike: float = 1.0) -> Tensor:
    """Returns the payoff of a lookback option with a fixed strike.

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(T, *)`, where, :math:`T` stands for the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return fn.relu(input.max(dim=0).values - strike)
    else:
        return fn.relu(strike - input.min(dim=0).values)


def american_binary_payoff(
    input: Tensor, call: bool = True, strike: float = 1.0
) -> Tensor:
    """
    Returns the payoff of an American binary option.

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(T, *)`, where, :math:`T` stands for the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return (input.max(dim=0).values >= strike).to(input)
    else:
        return (input.min(dim=0).values <= strike).to(input)


def european_binary_payoff(
    input: Tensor, call: bool = True, strike: float = 1.0
) -> Tensor:
    """
    Returns the payoff of a European binary option.

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(T, *)`, where, :math:`T` stands for the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return (input[-1] >= strike).to(input)
    else:
        return (input[-1] <= strike).to(input)


def exp_utility(input: Tensor, a: float = 1.0) -> Tensor:
    """Applies an exponential utility function.

    An exponential utility function is defined as:

    .. math ::

        u(x) = -\\exp(-a x) \,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float, default=1.0): The risk aversion coefficient of the exponential
            utility.

    Returns:
        torch.Tensor
    """
    return -torch.exp(-a * input)


def isoelastic_utility(input: Tensor, a: float = 0.5) -> Tensor:
    """Applies an isoelastic utility function.

    An isoelastic utility function is defined as:

    .. math ::

        u(x) = \\begin{cases}
        x^{1 - a} & a \\neq 1 \\\\
        \\log{x} & a = 1
        \\end{cases} \,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float, default=0.5): Relative risk aversion coefficient of the isoelastic
            utility.

    Returns:
        torch.Tensor
    """
    if a == 1.0:
        return torch.log(input)
    else:
        return torch.pow(input, exponent=1.0 - a)


def topp(input, p: float, dim: int = None, largest: bool = True):
    """Returns the largest `p * N` elements of the given input tensor,
    where `N` stands for the total number of elements in the input tensor.

    If `dim` is not given, the last dimension of the `input` is chosen.

    If `largest` is `False` then the smallest elements are returned.

    A namedtuple of `(values, indices)` is returned, where the `indices` are the indices
    of the elements in the original `input` tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): Quantile level.
        dim (int, optional): The dimension to sort along.
        largest (bool, default=True): Controls whether to return largest or smallest
            elements.

    Returns:
        torch.Tensor

    Examples:

        >>> input = torch.arange(1.0, 6.0)
        >>> input
        tensor([1., 2., 3., 4., 5.])
        >>> topp(input, 3 / 5)
        torch.return_types.topk(
        values=tensor([5., 4., 3.]),
        indices=tensor([4, 3, 2]))
    """
    if dim is None:
        return torch.topk(input, ceil(p * input.numel()), largest=largest)
    else:
        return torch.topk(input, ceil(p * input.size()[dim]), dim=dim, largest=largest)


def expected_shortfall(input: torch.Tensor, p: float, dim=None) -> torch.Tensor:
    """Returns the expected shortfall of the given input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): Quantile level.
        dim (int, optional): The dimension to sort along.
        largest (bool, default=True): Controls whether to return largest or smallest
            elements.

    Examples:

        >>> input = -torch.arange(1., 6.)
        >>> expected_shortfall(input, 3 / 5)
        tensor(4.)

    Returns:
        torch.Tensor
    """
    if dim is None:
        return -topp(input, p=p, largest=False).values.mean()
    else:
        return -topp(input, p=p, largest=False, dim=dim).values.mean(dim=dim)


def leaky_clamp(
    input: Tensor, min: Tensor = None, max: Tensor = None, clamped_slope: float = 0.01
) -> Tensor:
    """Leakily clamp all elements in `input` into the range :math:`[\\min, \\max]`.

    The bounds :math:`\\min` and :math:`\\max` can be tensors.

    See :class:`pfhedge.nn.LeakyClamp` for details.
    """
    x = input

    if min is not None:
        min = torch.as_tensor(min)
        x = torch.max(x, min + clamped_slope * (x - min))

    if max is not None:
        max = torch.as_tensor(max)
        x = torch.min(x, max + clamped_slope * (x - max))

    if min is not None and max is not None:
        x = torch.where(min <= max, x, (min + max) / 2)

    return x


def clamp(input: Tensor, min: Tensor = None, max: Tensor = None) -> Tensor:
    """Clamp all elements in `input` into the range :math:`[\\min, \\max]`.

    The bounds :math:`\\min` and :math:`\\max` can be tensors.

    See :class:`pfhedge.nn.Clamp` for details.
    """
    return leaky_clamp(input, min=min, max=max, clamped_slope=0.0)
