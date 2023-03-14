import contextlib
from typing import Optional
from typing import Union

import pytest
import torch
from torch.testing import assert_close

from pfhedge.nn.functional import bilerp
from pfhedge.nn.functional import box_muller
from pfhedge.nn.functional import clamp
from pfhedge.nn.functional import d1
from pfhedge.nn.functional import d2
from pfhedge.nn.functional import exp_utility
from pfhedge.nn.functional import expected_shortfall
from pfhedge.nn.functional import leaky_clamp
from pfhedge.nn.functional import pl
from pfhedge.nn.functional import realized_variance
from pfhedge.nn.functional import realized_volatility
from pfhedge.nn.functional import topp
from pfhedge.nn.functional import value_at_risk


def test_exp_utility(device: Optional[Union[str, torch.device]] = "cpu"):
    input = torch.tensor([-1.0, 0.0, 1.0], device=device)

    result = exp_utility(input, 1.0)
    expect = torch.tensor([-2.7183, -1.0000, -0.3679], device=device)
    assert_close(result, expect, atol=1e-4, rtol=1e-4)

    result = exp_utility(input, 2.0)
    expect = torch.tensor([-7.3891, -1.0000, -0.1353], device=device)
    assert_close(result, expect, atol=1e-4, rtol=1e-4)

    result = exp_utility(input, 0.0)
    expect = torch.tensor([-1.0000, -1.0000, -1.0000], device=device)
    assert_close(result, expect, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_exp_utility_gpu():
    test_exp_utility(device="cuda")


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("largest", [True, False])
def test_topp(p, largest, device: Optional[Union[str, torch.device]] = "cpu"):
    torch.manual_seed(42)

    input = torch.randn(100).to(device)
    k = int(p * 100)

    result = topp(input, p, largest=largest).values
    expect = torch.topk(input, k, largest=largest).values
    assert_close(result, expect)

    result = topp(input, p, largest=largest).indices
    expect = torch.topk(input, k, largest=largest).indices
    assert_close(result, expect)


@pytest.mark.gpu
@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("largest", [True, False])
def test_topp_gpu(p, largest):
    test_topp(p, largest, device="cuda")


def test_topp_error(device: Optional[Union[str, torch.device]] = "cpu"):
    with pytest.raises(RuntimeError):
        topp(torch.zeros(100).to(device), 1.1)
    with pytest.raises(RuntimeError):
        topp(torch.zeros(100).to(device), -0.1)


@pytest.mark.gpu
def test_topp_error_gpu():
    test_topp_error(device="cuda")


def test_expected_shortfall(device: Optional[Union[str, torch.device]] = "cpu"):
    input = torch.arange(1.0, 6.0).to(device)

    result = expected_shortfall(input, 3 / 5)
    expect = torch.tensor(-2.0).to(device)
    assert_close(result, expect)


@pytest.mark.gpu
def test_expected_shortfall_gpu():
    test_expected_shortfall(device="cuda")


def test_value_at_risk(device: Optional[Union[str, torch.device]] = "cpu"):
    input = -torch.arange(10.0).to(device)

    assert_close(value_at_risk(input, 0.0), -torch.tensor(9.0).to(device))
    assert_close(value_at_risk(input, 0.1), -torch.tensor(9.0).to(device))
    assert_close(value_at_risk(input, 0.2), -torch.tensor(8.0).to(device))
    assert_close(value_at_risk(input, 0.8), -torch.tensor(2.0).to(device))
    assert_close(value_at_risk(input, 0.9), -torch.tensor(1.0).to(device))
    assert_close(value_at_risk(input, 1.0), -torch.tensor(0.0).to(device))


@pytest.mark.gpu
def test_value_at_risk_gpu():
    test_value_at_risk(device="cuda")


def test_leaky_clamp(device: Optional[Union[str, torch.device]] = "cpu"):
    input = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], device=device)

    result = leaky_clamp(input, 0, 1, clamped_slope=0.1)
    expect = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.1], device=device)
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


@pytest.mark.gpu
def test_leaky_clamp_gpu():
    test_leaky_clamp(device="cuda")


def test_clamp_error_invalid_inverted_output(
    device: Optional[Union[str, torch.device]] = "cpu"
):
    input = torch.zeros(10).to(device)
    min = torch.zeros(10).to(device)
    max = torch.zeros(10).to(device)
    with pytest.raises(ValueError):
        _ = leaky_clamp(input, min, max, inverted_output="min")
    with pytest.raises(ValueError):
        _ = leaky_clamp(input, min, max, inverted_output="foo")
    with pytest.raises(ValueError):
        _ = clamp(input, min, max, inverted_output="min")
    with pytest.raises(ValueError):
        _ = clamp(input, min, max, inverted_output="foo")


@pytest.mark.gpu
def test_clamp_error_invalid_inverted_output_gpu():
    test_clamp_error_invalid_inverted_output(device="cuda")


def test_realized_variance(device: Optional[Union[str, torch.device]] = "cpu"):
    torch.manual_seed(42)

    log_return = 0.01 * torch.randn(2, 10).to(device)
    log_return[:, 0] = 0.0
    log_return -= log_return.mean(dim=-1, keepdim=True)
    input = log_return.cumsum(-1).exp()

    result = realized_variance(input, dt=1.0)
    expect = log_return[:, 1:].var(-1, unbiased=False)

    assert_close(result, expect)


@pytest.mark.gpu
def test_realized_variance_gpu():
    test_realized_variance(device="cuda")


def test_realized_volatility(device: Optional[Union[str, torch.device]] = "cpu"):
    torch.manual_seed(42)

    log_return = 0.01 * torch.randn(2, 10).to(device)
    log_return[:, 0] = 0.0
    log_return -= log_return.mean(dim=-1, keepdim=True)
    input = log_return.cumsum(-1).exp()

    result = realized_volatility(input, dt=1.0)
    expect = log_return[:, 1:].std(-1, unbiased=False)

    assert_close(result, expect)


@pytest.mark.gpu
def test_realized_volatility_gpu():
    test_realized_volatility(device="cuda")


def test_pl(device: Optional[Union[str, torch.device]] = "cpu"):
    N, T = 10, 20

    # pl = -payoff if unit = 0
    torch.manual_seed(42)
    spot = torch.randn((N, 1, T)).to(device).exp()
    unit = torch.zeros((N, 1, T)).to(device)
    payoff = torch.randn(N).to(device)
    result = pl(spot, unit, payoff=payoff)
    expect = -payoff
    assert_close(result, expect)

    # cost = 0
    torch.manual_seed(42)
    spot = torch.randn((N, 1, T)).to(device).exp()
    unit = torch.randn((N, 1, T)).to(device)
    result = pl(spot, unit)
    expect = ((spot[..., 1:] - spot[..., :-1]) * unit[..., :-1]).sum(-1).squeeze(1)
    assert_close(result, expect)

    # diff spot = 0, cost=0 -> value = 0
    torch.manual_seed(42)
    spot = torch.ones((N, 1, T)).to(device)
    unit = torch.randn((N, 1, T)).to(device)
    result = pl(spot, unit)
    expect = torch.zeros(N).to(device)
    assert_close(result, expect)

    # diff spot = 0, cost > 0 -> value = -cost
    torch.manual_seed(42)
    spot = torch.ones((N, 1, T)).to(device)
    unit = torch.randn((N, 1, T)).to(device)
    result = pl(spot, unit, cost=[1e-3], deduct_first_cost=False)
    expect = -1e-3 * ((unit[..., 1:] - unit[..., :-1]).abs() * spot[..., :-1]).sum(
        -1
    ).squeeze(1)
    assert_close(result, expect)

    torch.manual_seed(42)
    spot = torch.ones((N, 1, T)).to(device)
    unit = torch.randn((N, 1, T)).to(device)
    value0 = pl(spot, unit, cost=[1e-3], deduct_first_cost=False)
    value1 = pl(spot, unit, cost=[1e-3], deduct_first_cost=True)
    result = value1 - value0
    expect = -1e-3 * (unit[..., 0].abs() * spot[..., 1]).squeeze(1)
    assert_close(result, expect)


@pytest.mark.gpu
def test_pl_gpu():
    test_pl(device="cuda")


def test_pl_unmatched_shape(device: Optional[Union[str, torch.device]] = "cpu"):
    spot = torch.zeros((10, 1, 20)).to(device)
    unit = torch.zeros((10, 1, 20)).to(device)
    payoff = torch.zeros(10).to(device)
    with pytest.raises(RuntimeError):
        _ = pl(spot, unit[:-1])
    with pytest.raises(RuntimeError):
        _ = pl(spot, unit[:, :-1])
    with pytest.raises(RuntimeError):
        _ = pl(spot, unit, payoff=payoff[:-1])


@pytest.mark.gpu
def test_pl_unmatched_shape_gpu():
    test_pl_unmatched_shape(device="cuda")


def test_pl_additional_dim(device: Optional[Union[str, torch.device]] = "cpu"):
    N, M, T = 10, 30, 20

    # pnl = -payoff if unit = 0
    torch.manual_seed(42)
    spot = torch.randn((N, M, T)).to(device).exp()
    unit = torch.zeros((N, M, T)).to(device)
    payoff = torch.randn(N).to(device)
    result = pl(spot, unit, payoff=payoff)
    expect = -payoff
    assert_close(result, expect)


@pytest.mark.gpu
def test_pl_additional_dim_gpu():
    test_pl_additional_dim(device="cuda")


@pytest.mark.parametrize("log_moneyness", [-1.0, 0, 1.0])
@pytest.mark.parametrize("time_to_maturity", [1.0, 0, -1])
@pytest.mark.parametrize("volatility", [0.2, 0.0, -1])
def test_d1(
    log_moneyness: float,
    time_to_maturity: float,
    volatility: float,
    device: Optional[Union[str, torch.device]] = "cpu",
):
    with pytest.raises(
        ValueError
    ) if time_to_maturity < 0 or volatility < 0 else contextlib.nullcontext():
        result = d1(
            log_moneyness=torch.as_tensor([log_moneyness]).to(device),
            time_to_maturity=torch.as_tensor(time_to_maturity).to(device),
            volatility=torch.as_tensor(volatility).to(device),
        )
        assert not result.isnan()


@pytest.mark.gpu
@pytest.mark.parametrize("log_moneyness", [-1.0, 0, 1.0])
@pytest.mark.parametrize("time_to_maturity", [1.0, 0, -1])
@pytest.mark.parametrize("volatility", [0.2, 0.0, -1])
def test_d1_gpu(log_moneyness: float, time_to_maturity: float, volatility: float):
    test_d1(
        log_moneyness=log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        device="cuda",
    )


def test_d1_2(device: Optional[Union[str, torch.device]] = "cpu"):
    results = d1(
        log_moneyness=torch.as_tensor([-1.0, 0, 1.0]).to(device),
        time_to_maturity=torch.as_tensor([1.0, 0, 0]).to(device),
        volatility=torch.as_tensor([0.2, 0.2, 0.2]).to(device),
    )
    expected = torch.as_tensor([-4.9, 0, float("inf")]).to(device)
    assert_close(results, expected)
    with pytest.raises(ValueError):
        d1(
            log_moneyness=torch.as_tensor([-1.0, 0, 1.0]).to(device),
            time_to_maturity=torch.as_tensor([1.0, 0, -1]).to(device),
            volatility=torch.as_tensor([0.2, 0.2, 0.2]).to(device),
        )
    with pytest.raises(ValueError):
        d1(
            log_moneyness=torch.as_tensor([-1.0, 0, 1.0]).to(device),
            time_to_maturity=torch.as_tensor([1.0, 0, 0]).to(device),
            volatility=torch.as_tensor([0.2, 0.2, -1.0]).to(device),
        )


@pytest.mark.gpu
def test_d1_2_gpu():
    test_d1_2(device="cuda")


@pytest.mark.parametrize("log_moneyness", [-1.0, 0, 1.0])
@pytest.mark.parametrize("time_to_maturity", [1.0, 0, -1])
@pytest.mark.parametrize("volatility", [0.2, 0.0, -1])
def test_d2(
    log_moneyness: float,
    time_to_maturity: float,
    volatility: float,
    device: Optional[Union[str, torch.device]] = "cpu",
):
    with pytest.raises(
        ValueError
    ) if time_to_maturity < 0 or volatility < 0 else contextlib.nullcontext():
        result = d2(
            log_moneyness=torch.as_tensor([log_moneyness]).to(device),
            time_to_maturity=torch.as_tensor(time_to_maturity).to(device),
            volatility=torch.as_tensor(volatility).to(device),
        )
        assert not result.isnan()


@pytest.mark.gpu
@pytest.mark.parametrize("log_moneyness", [-1.0, 0, 1.0])
@pytest.mark.parametrize("time_to_maturity", [1.0, 0, -1])
@pytest.mark.parametrize("volatility", [0.2, 0.0, -1])
def test_d2_gpu(log_moneyness: float, time_to_maturity: float, volatility: float):
    test_d2(
        log_moneyness=log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        device="cuda",
    )


def test_d2_2(device: Optional[Union[str, torch.device]] = "cpu"):
    results = d2(
        log_moneyness=torch.as_tensor([-1.0, 0, 1.0]).to(device),
        time_to_maturity=torch.as_tensor([1.0, 0, 0]).to(device),
        volatility=torch.as_tensor([0.2, 0.2, 0.2]).to(device),
    )
    expected = torch.as_tensor([-5.1, 0, float("inf")]).to(device)
    assert_close(results, expected)
    with pytest.raises(ValueError):
        d2(
            log_moneyness=torch.as_tensor([-1.0, 0, 1.0]).to(device),
            time_to_maturity=torch.as_tensor([1.0, 0, -1]).to(device),
            volatility=torch.as_tensor([0.2, 0.2, 0.2]).to(device),
        )
    with pytest.raises(ValueError):
        d2(
            log_moneyness=torch.as_tensor([-1.0, 0, 1.0]).to(device),
            time_to_maturity=torch.as_tensor([1.0, 0, 0]).to(device),
            volatility=torch.as_tensor([0.2, 0.2, -1.0]).to(device),
        )


@pytest.mark.gpu
def test_d2_2_gpu():
    test_d2_2(device="cuda")


def test_bilerp(device: Optional[Union[str, torch.device]] = "cpu"):
    torch.manual_seed(42)

    i1 = torch.randn(2, 3).to(device)
    i2 = torch.randn(2, 3).to(device)
    i3 = torch.randn(2, 3).to(device)
    i4 = torch.randn(2, 3).to(device)

    # edge cases
    result = bilerp(i1, i2, i3, i4, 0.0, 0.0)
    assert_close(result, i1)
    result = bilerp(i1, i2, i3, i4, 1.0, 0.0)
    assert_close(result, i2)
    result = bilerp(i1, i2, i3, i4, 0.0, 1.0)
    assert_close(result, i3)
    result = bilerp(i1, i2, i3, i4, 1.0, 1.0)
    assert_close(result, i4)

    # w1 or w2 = 0 reduces to lerp
    result = bilerp(i1, i2, i3, i4, 0.1, 0.0)
    assert_close(result, torch.lerp(i1, i2, 0.1))
    result = bilerp(i1, i2, i3, i4, 0.0, 0.1)
    assert_close(result, torch.lerp(i1, i3, 0.1))

    result = bilerp(i1, i2, i3, i4, 0.5, 0.5)
    assert_close(result, (i1 + i2 + i3 + i4) / 4)


@pytest.mark.gpu
def test_bilerp_gpu():
    test_bilerp(device="cuda")


def test_box_muller(device: Optional[Union[str, torch.device]] = "cpu"):
    torch.manual_seed(42)

    # correct radius
    input1 = torch.rand(10).to(device)
    input2 = torch.rand(10).to(device)
    output1, output2 = box_muller(input1, input2)
    result = output1.square() + output2.square()
    expect = -2 * input1.clamp(min=1e-10).log()
    assert_close(result, expect)

    # correct angle
    input1 = torch.rand(10).to(device)
    input2 = torch.zeros(10).to(device)
    output1, output2 = box_muller(input1, input2)
    assert_close(output2, torch.zeros_like(output2))

    # no nan even when input1 is zero
    input1 = torch.zeros(10).to(device)
    input2 = torch.rand(10).to(device)
    output1, output2 = box_muller(input1, input2)
    assert not output1.isnan().any()
    assert not output2.isnan().any()


@pytest.mark.gpu
def test_box_muller_gpu():
    test_box_muller(device="cuda")
