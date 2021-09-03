import pytest
import torch
from torch.testing import assert_close

from pfhedge.nn.functional import clamp
from pfhedge.nn.functional import exp_utility
from pfhedge.nn.functional import expected_shortfall
from pfhedge.nn.functional import leaky_clamp
from pfhedge.nn.functional import realized_variance
from pfhedge.nn.functional import realized_volatility
from pfhedge.nn.functional import terminal_value
from pfhedge.nn.functional import topp


def test_exp_utility():
    input = torch.tensor([-1.0, 0.0, 1.0])

    result = exp_utility(input, 1.0)
    expect = torch.tensor([-2.7183, -1.0000, -0.3679])
    assert_close(result, expect, atol=1e-4, rtol=1e-4)

    result = exp_utility(input, 2.0)
    expect = torch.tensor([-7.3891, -1.0000, -0.1353])
    assert_close(result, expect, atol=1e-4, rtol=1e-4)

    result = exp_utility(input, 0.0)
    expect = torch.tensor([-1.0000, -1.0000, -1.0000])
    assert_close(result, expect, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("largest", [True, False])
def test_topp(p, largest):
    torch.manual_seed(42)

    input = torch.randn(100)
    k = int(p * 100)

    result = topp(input, p, largest=largest).values
    expect = torch.topk(input, k, largest=largest).values
    assert_close(result, expect)

    result = topp(input, p, largest=largest).indices
    expect = torch.topk(input, k, largest=largest).indices
    assert_close(result, expect)


def test_topp_error():
    with pytest.raises(RuntimeError):
        topp(torch.empty(100), 1.1)
    with pytest.raises(RuntimeError):
        topp(torch.empty(100), -0.1)


def test_expected_shortfall():
    input = torch.arange(1.0, 6.0)

    result = expected_shortfall(input, 3 / 5)
    expect = torch.tensor(-2.0)
    assert_close(result, expect)


def test_leaky_clamp():
    input = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])

    result = leaky_clamp(input, 0, 1, clamped_slope=0.1)
    expect = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.1])
    assert_close(result, expect)

    result = leaky_clamp(input, 0, 0, clamped_slope=0.01)
    expect = 0.01 * input
    assert_close(result, expect)

    result = leaky_clamp(input, 0, 1, clamped_slope=1)
    expect = input
    assert_close(result, expect)

    result = leaky_clamp(input, 1, 0, clamped_slope=0.01)
    expect = torch.full_like(result, 0.5)
    assert_close(result, expect)

    result = leaky_clamp(input, 1, 0, clamped_slope=0.01, inverted_output="max")
    expect = torch.full_like(result, 0.0)
    assert_close(result, expect)

    result = clamp(input, 1, 0, inverted_output="max")
    expect = torch.full_like(result, 0.0)
    assert_close(result, expect)

    result = leaky_clamp(input, 0, 1, clamped_slope=0.0)
    expect = clamp(input, 0, 1)
    assert_close(result, expect)


def test_clamp_error_invalid_inverted_output():
    input = torch.empty(10)
    min = torch.empty(10)
    max = torch.empty(10)
    with pytest.raises(ValueError):
        _ = leaky_clamp(input, min, max, inverted_output="min")
    with pytest.raises(ValueError):
        _ = leaky_clamp(input, min, max, inverted_output="foo")
    with pytest.raises(ValueError):
        _ = clamp(input, min, max, inverted_output="min")
    with pytest.raises(ValueError):
        _ = clamp(input, min, max, inverted_output="foo")


def test_realized_variance():
    torch.manual_seed(42)

    log_return = 0.01 * torch.randn(2, 10)
    log_return[:, 0] = 0.0
    log_return -= log_return.mean(dim=-1, keepdim=True)
    input = log_return.cumsum(-1).exp()

    result = realized_variance(input, dt=1.0)
    expect = log_return[:, 1:].var(-1, unbiased=False)

    assert_close(result, expect)


def test_realized_volatility():
    torch.manual_seed(42)

    log_return = 0.01 * torch.randn(2, 10)
    log_return[:, 0] = 0.0
    log_return -= log_return.mean(dim=-1, keepdim=True)
    input = log_return.cumsum(-1).exp()

    result = realized_volatility(input, dt=1.0)
    expect = log_return[:, 1:].std(-1, unbiased=False)

    assert_close(result, expect)


def test_terminal_value():
    N, T = 10, 20

    # pnl = -payoff if unit = 0
    torch.manual_seed(42)
    spot = torch.randn((N, T)).exp()
    unit = torch.zeros((N, T))
    payoff = torch.randn(N)
    result = terminal_value(spot, unit, payoff=payoff)
    expect = -payoff
    assert_close(result, expect)

    # cost = 0
    torch.manual_seed(42)
    spot = torch.randn((N, T)).exp()
    unit = torch.randn((N, T))
    result = terminal_value(spot, unit)
    expect = ((spot[..., 1:] - spot[..., :-1]) * unit[..., :-1]).sum(-1)
    assert_close(result, expect)

    # diff spot = 0, cost=0 -> value = 0
    torch.manual_seed(42)
    spot = torch.ones((N, T))
    unit = torch.randn((N, T))
    result = terminal_value(spot, unit)
    expect = torch.zeros(N)
    assert_close(result, expect)

    # diff spot = 0, cost > 0 -> value = -cost
    torch.manual_seed(42)
    spot = torch.ones((N, T))
    unit = torch.randn((N, T))
    result = terminal_value(spot, unit, cost=1e-3, deduct_first_cost=False)
    expect = -1e-3 * ((unit[..., 1:] - unit[..., :-1]).abs() * spot[..., :-1]).sum(-1)
    assert_close(result, expect)

    torch.manual_seed(42)
    spot = torch.ones((N, T))
    unit = torch.randn((N, T))
    value0 = terminal_value(spot, unit, cost=1e-3, deduct_first_cost=False)
    value1 = terminal_value(spot, unit, cost=1e-3, deduct_first_cost=True)
    result = value1 - value0
    expect = -1e-3 * unit[..., 0].abs() * spot[..., 1]
    assert_close(result, expect)


def test_terminal_value_unmatched_shape():
    spot = torch.empty((10, 20))
    unit = torch.empty((10, 20))
    payoff = torch.empty(10)
    with pytest.raises(RuntimeError):
        _ = terminal_value(spot, unit[:-1])
    with pytest.raises(RuntimeError):
        _ = terminal_value(spot, unit[:, :-1])
    with pytest.raises(RuntimeError):
        _ = terminal_value(spot, unit, payoff=payoff[:-1])


def test_terminal_value_additional_dim():
    N, M, T = 10, 30, 20

    # pnl = -payoff if unit = 0
    torch.manual_seed(42)
    spot = torch.randn((N, M, T)).exp()
    unit = torch.zeros((N, M, T))
    payoff = torch.randn(N, M)
    result = terminal_value(spot, unit, payoff=payoff)
    expect = -payoff
    assert_close(result, expect)
