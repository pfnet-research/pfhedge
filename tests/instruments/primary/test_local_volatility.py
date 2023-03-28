import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import LocalVolatilityStock


def test_local_volatility_zero_volatility(device: str = "cpu"):
    def zeros(time, spot):
        return torch.zeros_like(time)

    stock = LocalVolatilityStock(zeros).to(device)
    stock.simulate()

    assert_close(stock.spot, torch.ones_like(stock.spot))
    assert_close(stock.volatility, torch.zeros_like(stock.volatility))


@pytest.mark.gpu
def test_local_volatility_zero_volatility_gpu():
    test_local_volatility_zero_volatility(device="cuda")


def test_local_volatility(device: str = "cpu"):
    def zeros(time, spot):
        zero = torch.zeros_like(time)
        nonzero = torch.full_like(time, 0.2)
        return torch.where(time > 10 / 250, zero, nonzero)

    stock = LocalVolatilityStock(zeros).to(device)
    stock.simulate()

    result = stock.spot[:, 11:]
    expect = stock.spot[:, 10].unsqueeze(0).expand(-1, stock.spot[:, 11:].size(1))
    # TODO(masanorihirano): Remove check_stride once PyTorch 1.9.0 support is
    # no longer required. In PyTorch 1.10.0 and subsequent versions,
    # check_stride is set to False by default.
    assert_close(result, expect, check_stride=False)

    result = stock.volatility[:, 10:]
    expect = torch.zeros_like(stock.volatility[:, 10:])
    assert_close(result, expect)


@pytest.mark.gpu
def test_local_volatility_gpu():
    test_local_volatility(device="cuda")
