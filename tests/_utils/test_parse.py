import pytest
import torch
from torch.testing import assert_close
from pfhedge._utils.parse import parse_spot
from pfhedge._utils.parse import parse_volatility


def test_parse_spot():
    torch.manual_seed(42)

    spot = torch.randn(10).exp()
    strike = torch.randn(10).exp()
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

def test_parse_volatility():
    torch.manual_seed(42)

    volatility = torch.randn(10).exp()
    variance = volatility.square()

    result = parse_volatility(volatility=volatility)
    assert_close(result, volatility)
    result = parse_volatility(variance=variance)
    assert_close(result, volatility)

    with pytest.raises(ValueError):
        _ = parse_volatility()
    with pytest.raises(ValueError):
        _ = parse_volatility(spot=volatility)
