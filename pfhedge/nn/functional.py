from math import ceil
from typing import Optional
from typing import Union

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
        - input: :math:`(*, T)` where :math:`T` stands for the number of time steps
          and :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return fn.relu(input[..., -1] - strike)
    else:
        return fn.relu(strike - input[..., -1])


def lookback_payoff(input: Tensor, call: bool = True, strike: float = 1.0) -> Tensor:
    """Returns the payoff of a lookback option with a fixed strike.

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where :math:`T` stands for the number of time steps
          and :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return fn.relu(input.max(dim=-1).values - strike)
    else:
        return fn.relu(strike - input.min(dim=-1).values)


def american_binary_payoff(
    input: Tensor, call: bool = True, strike: float = 1.0
) -> Tensor:
    """Returns the payoff of an American binary option.

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where :math:`T` stands for the number of time steps
          and :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return (input.max(dim=-1).values >= strike).to(input)
    else:
        return (input.min(dim=-1).values <= strike).to(input)


def european_binary_payoff(
    input: Tensor, call: bool = True, strike: float = 1.0
) -> Tensor:
    """Returns the payoff of a European binary option.

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where :math:`T` stands for the number of time steps
          and :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return (input[..., -1] >= strike).to(input)
    else:
        return (input[..., -1] <= strike).to(input)


def exp_utility(input: Tensor, a: float = 1.0) -> Tensor:
    """Applies an exponential utility function.

    An exponential utility function is defined as:

    .. math ::

        u(x) = -\\exp(-a x) \\,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float, default=1.0): The risk aversion coefficient of the exponential
            utility.

    Returns:
        torch.Tensor
    """
    return -(-a * input).exp()


def isoelastic_utility(input: Tensor, a: float) -> Tensor:
    """Applies an isoelastic utility function.

    An isoelastic utility function is defined as:

    .. math ::

        u(x) = \\begin{cases}
        x^{1 - a} & a \\neq 1 \\\\
        \\log{x} & a = 1
        \\end{cases} \\,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float): Relative risk aversion coefficient of the isoelastic
            utility.

    Returns:
        torch.Tensor
    """
    if a == 1.0:
        return input.log()
    else:
        return input ** (1.0 - a)


def topp(input: Tensor, p: float, dim: Optional[int] = None, largest: bool = True):
    """Returns the largest :math:`p * N` elements of the given input tensor,
    where :math:`N` stands for the total number of elements in the input tensor.

    If ``dim`` is not given, the last dimension of the ``input`` is chosen.

    If ``largest`` is ``False`` then the smallest elements are returned.

    A namedtuple of ``(values, indices)`` is returned, where the ``indices``
    are the indices of the elements in the original ``input`` tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): Quantile level.
        dim (int, optional): The dimension to sort along.
        largest (bool, default=True): Controls whether to return largest or smallest
            elements.

    Returns:
        torch.Tensor

    Examples:

        >>> from pfhedge.nn.functional import topp
        >>>
        >>> input = torch.arange(1.0, 6.0)
        >>> input
        tensor([1., 2., 3., 4., 5.])
        >>> topp(input, 3 / 5)
        torch.return_types.topk(
        values=tensor([5., 4., 3.]),
        indices=tensor([4, 3, 2]))
    """
    if dim is None:
        return input.topk(ceil(p * input.numel()), largest=largest)
    else:
        return input.topk(ceil(p * input.size()[dim]), dim=dim, largest=largest)


def expected_shortfall(input: Tensor, p: float, dim: Optional[int] = None) -> Tensor:
    """Returns the expected shortfall of the given input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): Quantile level.
        dim (int, optional): The dimension to sort along.

    Examples:

        >>> from pfhedge.nn.functional import expected_shortfall
        >>>
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
    input: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    clamped_slope: float = 0.01,
) -> Tensor:
    """Leakily clamp all elements in ``input`` into the range :math:`[\\min, \\max]`.

    The bounds :math:`\\min` and :math:`\\max` can be tensors.

    See :class:`pfhedge.nn.LeakyClamp` for details.
    """
    x = input

    if min is not None:
        min = torch.as_tensor(min)
        x = x.maximum(min + clamped_slope * (x - min))

    if max is not None:
        max = torch.as_tensor(max)
        x = x.minimum(max + clamped_slope * (x - max))

    if min is not None and max is not None:
        x = x.where(min <= max, (min + max) / 2)

    return x


def clamp(
    input: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
) -> Tensor:
    """Clamp all elements in ``input`` into the range :math:`[\\min, \\max]`.

    The bounds :math:`\\min` and :math:`\\max` can be tensors.

    See :class:`pfhedge.nn.Clamp` for details.
    """
    return leaky_clamp(input, min=min, max=max, clamped_slope=0.0)


def realized_variance(input: Tensor, dt: Union[Tensor, float]) -> Tensor:
    """Returns the realized variance of the price.

    Realized variance :math:`\\sigma^2` of the stock price :math:`S` is defined as:

    .. math ::

        \\sigma^2 = \\frac{1}{T - 1} \\sum_{i = 1}^{T - 1}
        \\frac{1}{dt} \\log(S_{i + 1} / S_i)^2

    where :math:`T` is the number of time steps.

    .. note ::

        The mean of log return is assumed to be zero.

    Args:
        input (torch.Tensor): The input tensor.
        dt (torch.Tensor or float): The intervals of the time steps.

    Shape:
        - input: :math:`(*, T)` where :math:`T` stands for the number of time steps
          and :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    return input.log().diff(dim=-1).square().mean(dim=-1) / dt


def realized_volatility(input: Tensor, dt: Union[Tensor, float]) -> Tensor:
    """Returns the realized volatility of the price.
    It is square root of :func:`realized_variance`.

    Args:
        input (torch.Tensor): The input tensor.
        dt (torch.Tensor or float): The intervals of the time steps.

    Shape:
        - input: :math:`(*, T)` where :math:`T` stands for the number of time steps
          and :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    return realized_variance(input, dt=dt).sqrt()


def terminal_value(
    spot: Tensor,
    unit: Tensor,
    cost: float = 0.0,
    payoff: Optional[Tensor] = None,
    deduct_first_cost: bool = True,
):
    """Returns the terminal portfolio value.

    The terminal value of a hedger's portfolio is given by

    .. math::

        \\text{PL}(Z, \\delta, S) =
        - Z
        + \\sum_{i = 1}^{T - 1} \\delta_{i} (S_{i + 1} - S_i)
        - c \\sum_{i = 1}^{T - 1} |\\delta_{i + 1} - \\delta_{i}| S_{i + 1}

    where :math:`Z` is the payoff of the derivative, :math:`T` is the number of
    time steps, :math:`S` is the spot price, :math:`\\delta` is the signed number
    of shares held at each time step.

    Args:
        spot (torch.Tensor): The spot price of the underlying asset :math:`S`.
        unit (torch.Tensor): The signed number of shares of the underlying asset
            :math:`\\delta`.
        cost (float, default=0.0): The proportional transaction cost rate of
            the underlying asset :math:`c`.
        payoff (torch.Tensor, optional): The payoff of the derivative :math:`Z`.
        deduct_first_cost (bool, default=True): Whether to deduct the transaction
            cost of transacting stocks at the first time step.
            If ``True``, :math:`c |\delta_1| S_1` will be deducted from the above
            equation of the terminal value.

    Shape:
        - spot: :math:`(*, T)` where :math:`T` is the number of time steps.
        - unit: :math:`(*, T)`
        - payoff: :math:`(*)`
        - output: :math:`(*)`.

    Returns:
        torch.Tensor
    """
    if spot.size() != unit.size():
        raise RuntimeError(f"unmatched sizes: spot {spot.size()}, unit {unit.size()}")
    if payoff is not None and spot.size()[:-1] != payoff.size():
        raise RuntimeError(
            f"unmatched sizes: spot {spot.size()}, payoff {payoff.size()}"
        )

    value = unit[..., :-1].mul(spot.diff(dim=-1)).sum(-1)
    value += -cost * unit.diff(dim=-1).abs().mul(spot[..., 1:]).sum(-1)
    if payoff is not None:
        value -= payoff
    if deduct_first_cost:
        value -= cost * unit[..., 0].abs() * spot[..., 0]

    return value
