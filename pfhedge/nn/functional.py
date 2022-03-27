from math import ceil
from math import pi as kPI
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as fn
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.utils import broadcast_all

import pfhedge.autogreek as autogreek
from pfhedge._utils.typing import TensorOrScalar


def european_payoff(input: Tensor, call: bool = True, strike: float = 1.0) -> Tensor:
    """Returns the payoff of a European option.

    .. seealso::
        - :class:`pfhedge.instruments.EuropeanOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
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

    .. seealso::
        - :class:`pfhedge.instruments.LookbackOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
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

    .. seealso::
        - :class:`pfhedge.instruments.AmericanBinaryOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
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

    .. seealso::
        - :class:`pfhedge.instruments.EuropeanBinaryOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    if call:
        return (input[..., -1] >= strike).to(input)
    else:
        return (input[..., -1] <= strike).to(input)


def european_forward_start_payoff(
    input: Tensor, strike: float = 1.0, start_index: int = 0, end_index: int = -1
) -> Tensor:
    """Returns the payoff of a European forward start option.

    .. seealso::
        - :class:`pfhedge.instruments.EuropeanForwardStartOption`

    Args:
        input (torch.Tensor): The input tensor representing the price trajectory.
        strike (float, default=1.0): The strike price of the option.
        start_index (int, default=0): The time index at which the option starts.
        end_index (int, default=-1): The time index at which the option ends.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
        - output: :math:`(*)`

    Returns:
        torch.Tensor
    """
    return fn.relu(input[..., end_index] / input[..., start_index] - strike)


def exp_utility(input: Tensor, a: float = 1.0) -> Tensor:
    r"""Applies an exponential utility function.

    An exponential utility function is defined as:

    .. math::

        u(x) = -\exp(-a x) \,.

    Args:
        input (torch.Tensor): The input tensor.
        a (float, default=1.0): The risk aversion coefficient of the exponential
            utility.

    Returns:
        torch.Tensor
    """
    return -(-a * input).exp()


def isoelastic_utility(input: Tensor, a: float) -> Tensor:
    r"""Applies an isoelastic utility function.

    An isoelastic utility function is defined as:

    .. math::

        u(x) = \begin{cases}
        x^{1 - a} & a \neq 1 \\
        \log{x} & a = 1
        \end{cases} \,.

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
        return input.pow(1.0 - a)


def entropic_risk_measure(input: Tensor, a: float = 1.0) -> Tensor:
    """Returns the entropic risk measure.

    See :class:`pfhedge.nn.EntropicRiskMeasure` for details.
    """
    return (-exp_utility(input, a=a).mean(0)).log() / a


def topp(input: Tensor, p: float, dim: Optional[int] = None, largest: bool = True):
    """Returns the largest :math:`p * N` elements of the given input tensor,
    where :math:`N` stands for the total number of elements in the input tensor.

    If ``dim`` is not given, the last dimension of the ``input`` is chosen.

    If ``largest`` is ``False`` then the smallest elements are returned.

    A namedtuple of ``(values, indices)`` is returned, where the ``indices``
    are the indices of the elements in the original ``input`` tensor.

    .. seealso::
        - :func:`torch.topk`: Returns the ``k`` largest elements of the given input tensor
          along a given dimension.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The quantile level.
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
        return input.topk(ceil(p * input.size(dim)), dim=dim, largest=largest)


def expected_shortfall(input: Tensor, p: float, dim: Optional[int] = None) -> Tensor:
    """Returns the expected shortfall of the given input tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The quantile level.
        dim (int, optional): The dimension to sort along.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import expected_shortfall
        >>>
        >>> input = -torch.arange(10.0)
        >>> input
        tensor([-0., -1., -2., -3., -4., -5., -6., -7., -8., -9.])
        >>> expected_shortfall(input, 0.3)
        tensor(8.)
    """
    if dim is None:
        return -topp(input, p=p, largest=False).values.mean()
    else:
        return -topp(input, p=p, largest=False, dim=dim).values.mean(dim=dim)


def _min_values(input: Tensor, dim: Optional[int] = None) -> Tensor:
    return input.min() if dim is None else input.min(dim=dim).values


def _max_values(input: Tensor, dim: Optional[int] = None) -> Tensor:
    return input.max() if dim is None else input.max(dim=dim).values


def value_at_risk(input: Tensor, p: float, dim: Optional[int] = None) -> Tensor:
    """Returns the value at risk of the given input tensor.

    Note:
        If :math:`p \leq 1 / N`` with :math:`N` being the number of elements to sort,
        returns the smallest element in the tensor.
        If :math:`p > 1 - 1 / N``, returns the largest element in the tensor.

    Args:
        input (torch.Tensor): The input tensor.
        p (float): The quantile level.
        dim (int, optional): The dimension to sort along.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import value_at_risk
        >>>
        >>> input = -torch.arange(10.0)
        >>> input
        tensor([-0., -1., -2., -3., -4., -5., -6., -7., -8., -9.])
        >>> value_at_risk(input, 0.3)
        tensor(-7.)
    """
    n = input.numel() if dim is None else input.size(dim)

    if p <= 1 / n:
        output = _min_values(input, dim=dim)
    elif p > 1 - 1 / n:
        output = _max_values(input, dim=dim)
    else:
        q = (p - (1 / n)) / (1 - (1 / n))
        output = input.quantile(q, dim=dim)

    return output


def leaky_clamp(
    input: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    clamped_slope: float = 0.01,
    inverted_output: str = "mean",
) -> Tensor:
    r"""Leakily clamp all elements in ``input`` into the range :math:`[\min, \max]`.

    See :class:`pfhedge.nn.LeakyClamp` for details.
    """
    x = input

    if min is not None:
        min = torch.as_tensor(min).to(x)
        x = x.maximum(min + clamped_slope * (x - min))

    if max is not None:
        max = torch.as_tensor(max).to(x)
        x = x.minimum(max + clamped_slope * (x - max))

    if min is not None and max is not None:
        if inverted_output == "mean":
            y = (min + max) / 2
        elif inverted_output == "max":
            y = max
        else:
            raise ValueError("inverted_output must be 'mean' or 'max'.")
        x = x.where(min <= max, y)

    return x


def clamp(
    input: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    inverted_output: str = "mean",
) -> Tensor:
    r"""Clamp all elements in ``input`` into the range :math:`[\min, \max]`.

    See :class:`pfhedge.nn.Clamp` for details.
    """
    if inverted_output == "mean":
        output = leaky_clamp(input, min, max, clamped_slope=0.0, inverted_output="mean")
    elif inverted_output == "max":
        output = torch.clamp(input, min, max)
    else:
        raise ValueError("inverted_output must be 'mean' or 'max'.")
    return output


def realized_variance(input: Tensor, dt: TensorOrScalar) -> Tensor:
    r"""Returns the realized variance of the price.

    Realized variance :math:`\sigma^2` of the stock price :math:`S` is defined as:

    .. math::

        \sigma^2 = \frac{1}{T - 1} \sum_{i = 1}^{T - 1}
        \frac{1}{dt} \log(S_{i + 1} / S_i)^2

    where :math:`T` is the number of time steps.

    Note:
        The mean of log return is assumed to be zero.

    Args:
        input (torch.Tensor): The input tensor.
        dt (torch.Tensor or float): The intervals of the time steps.

    Shape:
        - input: :math:`(*, T)` where
          :math:`T` stands for the number of time steps and
          :math:`*` means any number of additional dimensions.
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
        - input: :math:`(*, T)` where
          :math:`T` stands for the number of time steps and
          :math:`*` means any number of additional dimensions.
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
) -> Tensor:
    r"""Returns the terminal portfolio value.

    The terminal value of a hedger's portfolio is given by

    .. math::

        \text{PL}(Z, \delta, S) =
        - Z
        + \sum_{i = 0}^{T - 2} \delta_{i - 1} (S_{i} - S_{i - 1})
        - c \sum_{i = 0}^{T - 1} |\delta_{i} - \delta_{i - 1}| S_{i}

    where :math:`Z` is the payoff of the derivative, :math:`T` is the number of
    time steps, :math:`S` is the spot price, :math:`\delta` is the signed number
    of shares held at each time step.
    We define :math:`\delta_0 = 0` for notational convenience.

    A hedger sells the derivative to its customer and
    obliges to settle the payoff at maturity.
    The dealer hedges the risk of this liability
    by trading the underlying instrument of the derivative.
    The resulting profit and loss is obtained by adding up the payoff to the
    customer, capital gains from the underlying asset, and the transaction cost.

    References:
        - Buehler, H., Gonon, L., Teichmann, J. and Wood, B., 2019.
          Deep hedging. Quantitative Finance, 19(8), pp.1271-1291.
          [arXiv:`1802.03042 <https://arxiv.org/abs/1802.03042>`_ [q-fin]]

    Args:
        spot (torch.Tensor): The spot price of the underlying asset :math:`S`.
        unit (torch.Tensor): The signed number of shares of the underlying asset
            :math:`\delta`.
        cost (float, default=0.0): The proportional transaction cost rate of
            the underlying asset :math:`c`.
        payoff (torch.Tensor, optional): The payoff of the derivative :math:`Z`.
        deduct_first_cost (bool, default=True): Whether to deduct the transaction
            cost of the stock at the first time step.
            If ``False``, :math:`- c |\delta_0| S_1` is omitted the above
            equation of the terminal value.

    Shape:
        - spot: :math:`(N, *, T)` where
          :math:`T` is the number of time steps and
          :math:`*` means any number of additional dimensions.
        - unit: :math:`(N, *, T)`
        - payoff: :math:`(N, *)`
        - output: :math:`(N, *)`.

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


def ncdf(input: Tensor) -> Tensor:
    r"""Returns a new tensor with the normal cumulative distribution function.

    .. math::
        \text{ncdf}(x) =
            \int_{-\infty}^x
            \frac{1}{\sqrt{2 \pi}} e^{-\frac{y^2}{2}} dy

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import ncdf
        >>>
        >>> input = torch.tensor([-1.0, 0.0, 10.0])
        >>> ncdf(input)
        tensor([0.1587, 0.5000, 1.0000])
    """
    return Normal(0.0, 1.0).cdf(input)


def npdf(input: Tensor) -> Tensor:
    r"""Returns a new tensor with the normal distribution function.

    .. math::
        \text{npdf}(x) = \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^2}{2}}

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.nn.functional import npdf
        >>>
        >>> input = torch.tensor([-1.0, 0.0, 10.0])
        >>> npdf(input)
        tensor([2.4197e-01, 3.9894e-01, 7.6946e-23])
    """
    return Normal(0.0, 1.0).log_prob(input).exp()


def d1(
    log_moneyness: TensorOrScalar,
    time_to_maturity: TensorOrScalar,
    volatility: TensorOrScalar,
) -> Tensor:
    r"""Returns :math:`d_1` in the Black-Scholes formula.

    .. math::
        d_1 = \frac{s}{\sigma \sqrt{t}} + \frac{\sigma \sqrt{t}}{2}

    where
    :math:`s` is the log moneyness,
    :math:`t` is the time to maturity, and
    :math:`\sigma` is the volatility.

    Note:
        Risk-free rate is set to zero.

    Args:
        log_moneyness (torch.Tensor or float): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor or float): Time to maturity of the derivative.
        volatility (torch.Tensor or float): Volatility of the underlying asset.

    Returns:
        torch.Tensor
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    if not (t >= 0).all():
        raise ValueError("all elements in time_to_maturity have to be non-negative")
    if not (v >= 0).all():
        raise ValueError("all elements in volatility have to be non-negative")
    variance = v * t.sqrt()
    output = s / variance + variance / 2
    # TODO(simaki): Replace zeros_like with 0.0 once https://github.com/pytorch/pytorch/pull/62084 is merged
    return output.where((s != 0).logical_or(variance != 0), torch.zeros_like(output))


def d2(
    log_moneyness: TensorOrScalar,
    time_to_maturity: TensorOrScalar,
    volatility: TensorOrScalar,
) -> Tensor:
    r"""Returns :math:`d_2` in the Black-Scholes formula.

    .. math::
        d_2 = \frac{s}{\sigma \sqrt{t}} - \frac{\sigma \sqrt{t}}{2}

    where
    :math:`s` is the log moneyness,
    :math:`t` is the time to maturity, and
    :math:`\sigma` is the volatility.

    Note:
        Risk-free rate is set to zero.

    Args:
        log_moneyness (torch.Tensor or float): Log moneyness of the underlying asset.
        time_to_maturity (torch.Tensor or float): Time to maturity of the derivative.
        volatility (torch.Tensor or float): Volatility of the underlying asset.

    Returns:
        torch.Tensor
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    if not (t >= 0).all():
        raise ValueError("all elements in time_to_maturity have to be non-negative")
    if not (v >= 0).all():
        raise ValueError("all elements in volatility have to be non-negative")
    variance = v * t.sqrt()
    output = s / variance - variance / 2
    # TODO(simaki): Replace zeros_like with 0.0 once https://github.com/pytorch/pytorch/pull/62084 is merged
    return output.where((s != 0).logical_or(variance != 0), torch.zeros_like(output))


def ww_width(
    gamma: Tensor, spot: Tensor, cost: TensorOrScalar, a: TensorOrScalar = 1.0
) -> Tensor:
    r"""Returns half-width of the no-transaction band for
    Whalley-Wilmott's hedging strategy.

    See :class:`pfhedge.nn.WhalleyWilmott` for details.

    Args:
        gamma (torch.Tensor): The gamma of the derivative,
        spot (torch.Tensor): The spot price of the underlier.
        cost (torch.Tensor or float): The cost rate of the underlier.
        a (torch.Tensor or float, default=1.0): Risk aversion parameter in exponential utility.

    Returns:
        torch.Tensor
    """
    return (cost * (3 / 2) * gamma.square() * spot / a).pow(1 / 3)


def svi_variance(
    input: TensorOrScalar,
    a: TensorOrScalar,
    b: TensorOrScalar,
    rho: TensorOrScalar,
    m: TensorOrScalar,
    sigma: TensorOrScalar,
) -> Tensor:
    r"""Returns variance in the SVI model.

    See :class:`pfhedge.nn.SVIVariance` for details.

    Args:
        input (torch.Tensor or float): Log strike of the underlying asset.
            That is, :math:`k = \log(K / S)` for spot :math:`S` and strike :math:`K`.
        a (torch.Tensor or float): The parameter :math:`a`.
        b (torch.Tensor or float): The parameter :math:`b`.
        rho (torch.Tensor or float): The parameter :math:`\rho`.
        m (torch.Tensor or float): The parameter :math:`m`.
        sigma (torch.Tensor or float): The parameter :math:`s`.

    Returns:
        torch.Tensor
    """
    k_m = torch.as_tensor(input - m)  # k - m
    return a + b * (rho * k_m + (k_m.square() + sigma ** 2).sqrt())


def bilerp(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    input4: Tensor,
    weight1: TensorOrScalar,
    weight2: TensorOrScalar,
) -> Tensor:
    r"""Does a bilinear interpolation of four tensors based on a scalar or tensor weights and
    returns the resulting tensor.

    The output is given by

    .. math::
        \text{output}_i
        & = (1 - w_1) (1 - w_2) \cdot \text{input1}_i
        + w_1 (1 - w_2) \cdot \text{input2}_i \\
        & \quad + (1 - w_1) w_2 \cdot \text{input3}_i
        + w_1 w_2 \cdot \text{input4}_i ,

    where :math:`w_1` and :math:`w_2` are the weights.

    The shapes of inputs must be broadcastable.
    If ``weight`` is a tensor, then the shapes of ``weight`` must also be broadcastable.

    Args:
        input1 (torch.Tensor): The input tensor.
        input2 (torch.Tensor): The input tensor.
        input3 (torch.Tensor): The input tensor.
        input4 (torch.Tensor): The input tensor.
        weight1 (float or torch.Tensor): The weight tensor.
        weight2 (float or torch.Tensor): The weight tensor.

    Returns:
        torch.Tensor
    """
    lerp1 = torch.lerp(input1, input2, weight1)
    lerp2 = torch.lerp(input3, input4, weight1)
    return torch.lerp(lerp1, lerp2, weight2)


def _bs_theta_gamma_relation(gamma: Tensor, spot: Tensor, volatility: Tensor) -> Tensor:
    # theta = -(1/2) * vola^2 * spot^2 * gamma
    # by Black-Scholes formula
    return -gamma * volatility.square() * spot.square() / 2


def _bs_vega_gamma_relation(
    gamma: Tensor, spot: Tensor, time_to_maturity: Tensor, volatility: Tensor
) -> Tensor:
    # vega = vola * spot^2 * time * gamma
    # in Black-Scholes model
    # See Chapter 5 Appendix A, Bergomi "Stochastic volatility modeling"
    return gamma * volatility * spot.square() * time_to_maturity


def bs_european_price(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar = 1.0,
    call: bool = True,
) -> Tensor:
    """Returns Black-Scholes price of a European option.

    See :func:`pfhedge.nn.BSEuropeanOption.price` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    spot = s.exp() * strike
    price = spot * ncdf(d1(s, t, v)) - strike * ncdf(d2(s, t, v))
    price = price + strike * (1 - s.exp()) if not call else price  # put-call parity

    return price


def bs_european_delta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
) -> Tensor:
    """Returns Black-Scholes delta of a European option.

    See :func:`pfhedge.nn.BSEuropeanOption.delta` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    delta = ncdf(d1(s, t, v))
    delta = delta - 1 if not call else delta  # put-call parity

    return delta


def bs_european_gamma(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    """Returns Black-Scholes gamma of a European option.

    See :func:`pfhedge.nn.BSEuropeanOption.gamma` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = strike * s.exp()
    numerator = npdf(d1(s, t, v))
    denominator = spot * v * t.sqrt()
    output = numerator / denominator
    return torch.where(
        (numerator == 0).logical_and(denominator == 0), torch.zeros_like(output), output
    )


def bs_european_vega(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes vega of a European option.

    See :func:`pfhedge.nn.BSEuropeanOption.vega` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    price = strike * s.exp()
    return npdf(d1(s, t, v)) * price * t.sqrt()


def bs_european_theta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes theta of a European option.

    See :func:`pfhedge.nn.BSEuropeanOption.theta` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    price = strike * s.exp()
    numerator = -npdf(d1(s, t, v)) * price * v
    denominator = 2 * t.sqrt()
    output = numerator / denominator
    return torch.where(
        (numerator == 0).logical_and(denominator == 0), torch.zeros_like(output), output
    )


def bs_european_binary_price(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
) -> Tensor:
    """Returns Black-Scholes price of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.price` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    price = ncdf(d2(s, t, v))
    price = 1.0 - price if not call else price  # put-call parity

    return price


def bs_european_binary_delta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    """Returns Black-Scholes delta of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.delta` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    spot = s.exp() * strike

    numerator = npdf(d2(s, t, v))
    denominator = spot * v * t.sqrt()
    delta = numerator / denominator
    delta = torch.where(
        (numerator == 0).logical_and(denominator == 0), torch.zeros_like(delta), delta
    )
    delta = -delta if not call else delta  # put-call parity

    return delta


def bs_european_binary_gamma(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    """Returns Black-Scholes gamma of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.gamma` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike

    d2_tensor = d2(s, t, v)
    w = volatility * time_to_maturity.square()

    gamma = -npdf(d2_tensor).div(w * spot.square()) * (1 + d2_tensor.div(w))

    gamma = -gamma if not call else gamma  # put-call parity

    return gamma


def bs_european_binary_vega(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    """Returns Black-Scholes vega of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.vega` for details.
    """
    gamma = bs_european_binary_gamma(
        log_moneyness=log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        call=call,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_vega_gamma_relation(
        gamma, spot=spot, time_to_maturity=time_to_maturity, volatility=volatility
    )


def bs_european_binary_theta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    """Returns Black-Scholes theta of a European binary option.

    See :func:`pfhedge.nn.BSEuropeanBinaryOption.theta` for details.
    """
    gamma = bs_european_binary_gamma(
        log_moneyness=log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        call=call,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_theta_gamma_relation(gamma, spot=spot, volatility=volatility)


def bs_american_binary_price(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
) -> Tensor:
    """Returns Black-Scholes price of an American binary option.

    See :func:`pfhedge.nn.BSAmericanBinaryOption.price` for details.
    """
    # This formula is derived using the results in Section 7.3.3 of Shreve's book.
    # Price is I_2 - I_4 where the interval of integration is [k --> -inf, b].
    # By this substitution we get N([log(S(0) / K) + ...] / sigma T) --> 1.

    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    p = ncdf(d2(s, t, v)) + s.exp() * ncdf(d1(s, t, v))

    return p.where(max_log_moneyness < 0, torch.ones_like(p))


def bs_american_binary_delta(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes delta of an American binary option.

    See :func:`pfhedge.nn.BSAmericanBinaryOption.delta` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike

    d1_tensor = d1(s, t, v)
    d2_tensor = d2(s, t, v)
    w = v * t.sqrt()

    # ToDo: fix 0/0 issue
    p = (
        npdf(d2_tensor).div(spot * w)
        + ncdf(d1_tensor).div(strike)
        + npdf(d1_tensor).div(strike * w)
    )
    return p.where(max_log_moneyness < 0, torch.zeros_like(p))


def bs_american_binary_gamma(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes gamma of an American binary option.

    See :func:`pfhedge.nn.BSAmericanBinaryOption.gamma` for details.
    """
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike

    d1_tensor = d1(s, t, v)
    d2_tensor = d2(s, t, v)
    w = v * t.sqrt()

    p = (
        -npdf(d2_tensor).div(spot.square() * w)
        - d2_tensor * npdf(d2_tensor).div(spot.square() * w.square())
        + npdf(d1_tensor).div(spot * strike * w)
        - d1_tensor * npdf(d1_tensor).div(spot * strike * w.square())
    )
    return p.where(max_log_moneyness < 0, torch.zeros_like(p))


def bs_american_binary_vega(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes vega of an American binary option.

    See :func:`pfhedge.nn.BSAmericanBinaryOption.vega` for details.
    """
    gamma = bs_american_binary_gamma(
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_vega_gamma_relation(
        gamma, spot=spot, time_to_maturity=time_to_maturity, volatility=volatility
    )


def bs_american_binary_theta(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes theta of an American binary option.

    See :func:`pfhedge.nn.BSAmericanBinaryOption.theta` for details.
    """
    gamma = bs_american_binary_gamma(
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_theta_gamma_relation(gamma, spot=spot, volatility=volatility)


def bs_lookback_price(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes price of a lookback option.

    See :func:`pfhedge.nn.BSLookbackOption.price` for details.
    """
    s, m, t, v = map(
        torch.as_tensor,
        (log_moneyness, max_log_moneyness, time_to_maturity, volatility),
    )

    spot = s.exp() * strike
    max = m.exp() * strike
    d1_value = d1(s, t, v)
    d2_value = d2(s, t, v)
    m1 = d1(s - m, t, v)  # d' in the paper
    m2 = d2(s - m, t, v)

    # when max < strike
    price_0 = spot * (
        ncdf(d1_value) + v * t.sqrt() * (d1_value * ncdf(d1_value) + npdf(d1_value))
    ) - strike * ncdf(d2_value)
    # when max >= strike
    price_1 = (
        spot * (ncdf(m1) + v * t.sqrt() * (m1 * ncdf(m1) + npdf(m1)))
        - strike
        + max * (1 - ncdf(m2))
    )

    return torch.where(max < strike, price_0, price_1)


def bs_lookback_delta(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes delta of a lookback option.

    See :func:`pfhedge.nn.BSLookbackOption.delta` for details.
    """
    # TODO(simaki): Calculate analytically
    return autogreek.delta(
        bs_lookback_price,
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )


def bs_lookback_gamma(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes gamma of a lookback option.

    See :func:`pfhedge.nn.BSLookbackOption.gamma` for details.
    """
    # TODO(simaki): Calculate analytically
    return autogreek.gamma(
        bs_lookback_price,
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )


def bs_lookback_vega(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes vega of a lookback option.

    See :func:`pfhedge.nn.BSLookbackOption.vega` for details.
    """
    gamma = bs_lookback_gamma(
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_vega_gamma_relation(
        gamma, spot=spot, time_to_maturity=time_to_maturity, volatility=volatility
    )


def bs_lookback_theta(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    """Returns Black-Scholes theta of a lookback option.

    See :func:`pfhedge.nn.BSLookbackOption.theta` for details.
    """
    gamma = bs_lookback_gamma(
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_theta_gamma_relation(gamma, spot=spot, volatility=volatility)


def box_muller(
    input1: Tensor, input2: Tensor, epsilon: float = 1e-10
) -> Tuple[Tensor, Tensor]:
    r"""Returns two tensors obtained by applying Box-Muller transformation to two input tensors.

    .. math::
        & \mathrm{output1}_i
            = \sqrt{- 2 \log (\mathrm{input1}_i)} \cos(2 \pi \cdot \mathrm{input2}_i) , \\
        & \mathrm{output2}_i
            = \sqrt{- 2 \log (\mathrm{input1}_i)} \sin(2 \pi \cdot \mathrm{input2}_i) .

    Args:
        input1 (torch.Tensor): The first input tensor.
        input2 (torch.Tensor): The second input tensor.
        epsilon (float, default=1e-10): A small constant to avoid evaluating :math:`\log(0)`.
            The tensor ``input1`` will be clamped with this value being the minimum.

    Returns:
        (torch.Tensor, torch.Tensor)
    """
    radius = (-2 * input1.clamp(min=epsilon).log()).sqrt()
    angle = 2 * kPI * input2
    output1 = radius * angle.cos()
    output2 = radius * angle.sin()
    return output1, output2
