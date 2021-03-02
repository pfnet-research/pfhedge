from math import ceil

import torch
import torch.nn.functional as fn


def european_payoff(input: torch.Tensor, call=True, strike=1.0) -> torch.Tensor:
    """
    Returns the payoff of a European option.

    Parameters
    ----------
    - input : Tensor, shape (N_STEPS, *)
        The input tensor representing the price trajectory.
        Here, `N_STEPS` stands for the number of time steps and
        `*` means any number of additional dimensions.
    - call : bool, default True
        Specify whether the option is call or put.
    - strike : float, default 1.0
        The strike price of the option.

    Returns
    -------
    payoff : Tensor, shape (*)
    """
    if call:
        return fn.relu(input[-1] - strike)
    else:
        return fn.relu(strike - input[-1])


def lookback_payoff(input: torch.Tensor, call=True, strike=1.0):
    """
    Returns the payoff of a lookback option with a fixed strike.

    Parameters
    ----------
    - input : Tensor, shape (N_STEPS, *)
        The input tensor representing the price trajectory.
        Here, `N_STEPS` stands for the number of time steps and
        `*` means any number of additional dimensions.
    - call : bool, default True
        Specify whether the option is call or put.
    - strike : float, default 1.0
        The strike price of the option.

    Returns
    -------
    payoff : Tensor, shape (*)
    """
    if call:
        return fn.relu(input.max(dim=0).values - strike)
    else:
        return fn.relu(strike - input.min(dim=0).values)


def american_binary_payoff(input: torch.Tensor, call=True, strike=1.0):
    """
    Returns the payoff of an American binary option.

    Parameters
    ----------
    - input : Tensor, shape (N_STEPS, *)
        The input tensor representing the price trajectory.
        Here, `N_STEPS` stands for the number of time steps and
        `*` means any number of additional dimensions.
    - call : bool, default True
        Specify whether the option is call or put.
    - strike : float, default 1.0
        The strike price of the option.

    Returns
    -------
    payoff : Tensor, shape (*)
    """
    if call:
        return (input.max(dim=0).values >= strike).to(input)
    else:
        return (input.min(dim=0).values <= strike).to(input)


def european_binary_payoff(input: torch.Tensor, call=True, strike=1.0):
    """
    Returns the payoff of a European binary option.

    Parameters
    ----------
    - input : Tensor, shape (N_STEPS, *)
        The input tensor representing the price trajectory.
        Here, `N_STEPS` stands for the number of time steps and
        `*` means any number of additional dimensions.
    - call : bool, default True
        Specify whether the option is call or put.
    - strike : float, default 1.0
        The strike price of the option.

    Returns
    -------
    payoff : Tensor, shape (*)
    """
    if call:
        return (input[-1] >= strike).to(input)
    else:
        return (input[-1] <= strike).to(input)


def exp_utility(input: torch.Tensor, a=1.0) -> torch.Tensor:
    """
    Applies an exponential utility function.

    An exponential utility function is defined as:

        u(input) = -exp(-a * input)

    Parameters
    ----------
    - input : Tensor, shape (*)
        The input tensor.
    - a : float, default 1.0
        The risk aversion coefficient of the exponential utility.

    Returns
    -------
    exp_utility : Tensor, shape (*)
    """
    return -torch.exp(-a * input)


def isoelastic_utility(input: torch.Tensor, a=0.5):
    """
    Applies an isoelastic utility function.

    An isoelastic utility function is defined as:

        u(x) = x ** (1 - a) if a != 1.0
               log(x) if a == 1.0

    Parameters
    ----------
    - input : Tensor, shape (*)
        The input tensor.
    - a : float, default 1.0
        Relative risk aversion coefficient of the isoelastic utility.

    Returns
    -------
    isoelastic_utility : Tensor, shape (*)
    """
    if a == 1.0:
        return torch.log(input)
    else:
        return torch.pow(input, exponent=1.0 - a)


def topp(input, p, dim=None, largest=True) -> (torch.Tensor, torch.LongTensor):
    """
    Returns the largest `p * N` elements of the given input tensor,
    where `N` stands for the total number of elements in the input tensor.

    If `largest` is `False` then the smallest elements are returned.

    Parameters
    ----------
    - input : Tensor
        The input tensor.
    - p : float
        Quantile level.
    - dim : int, optional
        The dimension to sort along.
    - largest : bool, default True
        Controls whether to return largest or smallest elements.

    Examples
    --------
    >>> input = torch.arange(1., 6.)
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
    """
    Returns the expected shortfall of the given input tensor.

    Parameters
    ----------
    - input : Tensor, shape (*)
        Input tensor.
    - p : float
        Quantile level.
    - dim : int
        The dimension or dimensions to reduce.

    Examples
    --------
    >>> input = -torch.arange(1., 6.)
    >>> expected_shortfall(input, 3 / 5)
    tensor(4.)

    Returns
    -------
    expected_shortfall : Tensor
    """
    if dim is None:
        return -topp(input, p=p, largest=False).values.mean()
    else:
        return -topp(input, p=p, largest=False, dim=dim).values.mean(dim=dim)


def leaky_clamp(
    input, min_value=None, max_value=None, clamped_slope=0.01
) -> torch.Tensor:
    """
    Clamp all elements in the input tensor into the range [`min_value`, `max_value`]
    and return a resulting tensor.
    The bounds `min_value` and `max_value` can be tensors.

    See `pfhedge.nn.LeakyClamp` for details.
    """
    x = input

    if min_value is not None:
        min_value = torch.as_tensor(min_value)
        x = torch.max(input, min_value + clamped_slope * (x - min_value))

    if max_value is not None:
        max_value = torch.as_tensor(max_value)
        x = torch.min(x, max_value + clamped_slope * (x - max_value))

    if min_value is not None and max_value is not None:
        x = torch.where(min_value <= max_value, x, (min_value + max_value) / 2)

    return x


def clamp(input, min_value=None, max_value=None) -> torch.Tensor:
    """
    Clamp all elements in the input tensor into the range [`min_value`, `max_value`]
    and return a resulting tensor.
    The bounds `min_value` and `max_value` can be tensors.

    See `pfhedge.nn.Clamp` for details.
    """
    return leaky_clamp(
        input, min_value=min_value, max_value=max_value, clamped_slope=0.0
    )
