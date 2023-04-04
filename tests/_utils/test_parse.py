import pytest
import torch
from torch.testing import assert_close

from pfhedge._utils.parse import parse_spot
from pfhedge._utils.parse import parse_time_to_maturity
from pfhedge._utils.parse import parse_volatility


def test_parse_spot(device: str = "cpu"):
    torch.manual_seed(42)

    spot = torch.randn(10).to(device).exp()
    strike = torch.randn(10).to(device).exp()
    moneyness = spot / strike
    log_moneyness = moneyness.log()

    result = parse_spot(spot=spot)
    assert_close(result, spot)

    result = parse_spot(moneyness=moneyness, strike=strike)
    assert_close(result, spot)

    result = parse_spot(moneyness=moneyness, strike=strike)
    assert_close(result, spot)

    with pytest.raises(ValueError):
        _ = parse_spot()
    with pytest.raises(ValueError):
        _ = parse_spot(log_moneyness=moneyness)
    with pytest.raises(ValueError):
        _ = parse_spot(log_moneyness=log_moneyness)


@pytest.mark.gpu
def test_parse_spot_gpu():
    test_parse_spot(device="cuda")


def test_parse_volatility(device: str = "cpu"):
    torch.manual_seed(42)

    volatility = torch.randn(10).to(device).exp()
    variance = volatility.square()

    result = parse_volatility(volatility=volatility)
    assert_close(result, volatility)
    result = parse_volatility(variance=variance)
    assert_close(result, volatility)

    with pytest.raises(ValueError):
        _ = parse_volatility()
    with pytest.raises(ValueError):
        _ = parse_volatility(spot=volatility)


@pytest.mark.gpu
def test_parse_volatility_gpu():
    test_parse_volatility(device="cuda")


def test_parse_time_to_maturity(device: str = "cpu"):
    torch.manual_seed(42)

    time_to_maturity = torch.randn(10).to(device).exp()

    result = parse_time_to_maturity(time_to_maturity=time_to_maturity)
    assert_close(result, time_to_maturity)

    with pytest.raises(ValueError):
        _ = parse_time_to_maturity()
    with pytest.raises(ValueError):
        _ = parse_time_to_maturity(spot=time_to_maturity)


@pytest.mark.gpu
def test_parse_time_to_maturity_gpu():
    test_parse_time_to_maturity(device="cuda")
