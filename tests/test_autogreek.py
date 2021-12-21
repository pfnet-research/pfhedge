import torch
from torch.testing import assert_close

import pfhedge.autogreek as autogreek


def test_vega():
    def pricer(volatility, coef):
        return coef * volatility.pow(2)

    torch.manual_seed(42)
    volatility = torch.randn(10).exp()  # make it positive
    coef = torch.randn(10)
    result = autogreek.vega(pricer, volatility=volatility, coef=coef)
    expect = 2 * coef * volatility
    assert_close(result, expect)

    torch.manual_seed(42)
    variance = torch.randn(10).exp()  # make it positive
    coef = torch.randn(10)
    result = autogreek.vega(pricer, variance=variance, coef=coef)
    expect = 2 * coef * variance.sqrt()
    assert_close(result, expect)


def test_theta():
    def pricer(time_to_maturity, coef):
        return coef * time_to_maturity.pow(2)

    torch.manual_seed(42)
    time_to_maturity = torch.randn(10).exp()  # make it positive
    coef = torch.randn(10)
    result = autogreek.theta(pricer, time_to_maturity=time_to_maturity, coef=coef)
    expect = -2 * coef * time_to_maturity
    assert_close(result, expect)
