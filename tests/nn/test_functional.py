import pytest
import torch
from torch.testing import assert_close

from pfhedge.nn.functional import clamp
from pfhedge.nn.functional import exp_utility
from pfhedge.nn.functional import expected_shortfall
from pfhedge.nn.functional import leaky_clamp
from pfhedge.nn.functional import topp


def test_exp_utility():
    x = torch.tensor([-1.0, 0.0, 1.0])

    result = exp_utility(x, 1.0)
    expect = torch.tensor([-2.7183, -1.0000, -0.3679])
    assert_close(result, expect, atol=1e-4, rtol=1e-4)

    result = exp_utility(x, 2.0)
    expect = torch.tensor([-7.3891, -1.0000, -0.1353])
    assert_close(result, expect, atol=1e-4, rtol=1e-4)

    result = exp_utility(x, 0.0)
    expect = torch.tensor([-1.0000, -1.0000, -1.0000])
    assert_close(result, expect, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("largest", [True, False])
def test_topp(p, largest):
    torch.manual_seed(42)

    x = torch.randn(100)
    k = int(p * 100)

    result = topp(x, p, largest=largest).values
    expect = torch.topk(x, k, largest=largest).values
    assert_close(result, expect)

    result = topp(x, p, largest=largest).indices
    expect = torch.topk(x, k, largest=largest).indices
    assert_close(result, expect)


def test_topp_error():
    with pytest.raises(RuntimeError):
        topp(torch.empty(100), 1.1)
    with pytest.raises(RuntimeError):
        topp(torch.empty(100), -0.1)


def test_expected_shortfall():
    x = torch.arange(1.0, 6.0)

    result = expected_shortfall(x, 3 / 5)
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

    result = leaky_clamp(input, 0, 1, clamped_slope=0.0)
    expect = clamp(input, 0, 1)
    assert_close(result, expect)
