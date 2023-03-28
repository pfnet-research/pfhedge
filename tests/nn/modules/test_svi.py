import pytest
import torch
from torch.nn.functional import relu
from torch.testing import assert_allclose

from pfhedge.nn import SVIVariance


def test_svi(device: str = "cpu"):
    input = torch.linspace(-1.0, 1.0, 10).to(device)

    # for b = 0, output = a
    m = SVIVariance(a=1.0, b=0, rho=0.1, m=0.2, sigma=0.3).to(device)
    result = m(input)
    expect = torch.full_like(result, m.a)
    assert_allclose(result, expect)

    # for sigma = 0 and rho = -1, output = a + 2 b relu(x)
    m = SVIVariance(a=1.0, b=0, rho=-1, m=0.0, sigma=0.3).to(device)
    result = m(input)
    expect = m.a + 2 * m.b * relu(input - m.m)
    assert_allclose(result, expect)

    # m translates
    m0 = SVIVariance(a=0.1, b=0.2, rho=0.3, m=0.0, sigma=0.5).to(device)
    m1 = SVIVariance(a=0.1, b=0.2, rho=0.3, m=0.4, sigma=0.5).to(device)
    result = m0(input)
    expect = m1(input + m1.m)
    assert_allclose(result, expect)


@pytest.mark.gpu
def test_svi_gpu():
    test_svi(device="cuda")


def test_svi_repr():
    m = SVIVariance(a=1.0, b=0, rho=0.1, m=0.2, sigma=0.3)
    assert repr(m) == "SVIVariance(a=1.0, b=0, rho=0.1, m=0.2, sigma=0.3)"
