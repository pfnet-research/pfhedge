import torch
from torch.testing import assert_close

from pfhedge.instruments import LocalVolatilityStock


def test_local_volatility_zero_volatility():
    def zeros(time, spot):
        return torch.zeros_like(time)

    stock = LocalVolatilityStock(zeros)
    stock.simulate()

    assert_close(stock.spot, torch.ones_like(stock.spot))
    assert_close(stock.volatility, torch.zeros_like(stock.volatility))


def test_local_volatility():
    def zeros(time, spot):
        zero = torch.zeros_like(time)
        nonzero = torch.full_like(time, 0.2)
        return torch.where(time > 10 / 250, zero, nonzero)

    stock = LocalVolatilityStock(zeros)
    stock.simulate()

    result = stock.spot[:, 11:]
    expect = stock.spot[:, 10].unsqueeze(0).expand(-1, stock.spot[:, 11:].size(1))
    assert_close(result, expect)

    result = stock.volatility[:, 10:]
    expect = torch.zeros_like(stock.volatility[:, 10:])
    assert_close(result, expect)
