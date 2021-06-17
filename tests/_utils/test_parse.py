import pytest
import torch
from torch.testing import assert_equal

from pfhedge._utils.parse import parse_spot


def test_parse():
    spot = torch.arange(3.0)
    strike = torch.full_like(spot, 2.0)
    moneyness = spot / strike
    log_moneyness = moneyness.log()

    result = parse_spot(spot=spot)
    assert_equal(result, spot)

    result = parse_spot(moneyness=moneyness, strike=strike)
    assert_equal(result, spot)

    result = parse_spot(moneyness=moneyness, strike=strike)
    assert_equal(result, spot)

    with pytest.raises(ValueError):
        _ = parse_spot()

    with pytest.raises(ValueError):
        _ = parse_spot(log_moneyness=moneyness)

    with pytest.raises(ValueError):
        _ = parse_spot(log_moneyness=log_moneyness)
