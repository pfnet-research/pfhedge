from math import sqrt

import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import HestonStock
from pfhedge.instruments import VarianceSwap

cls = VarianceSwap


class TestVarianceSwap:
    def test_payoff(self, device: str = "cpu"):
        derivative = VarianceSwap(BrownianStock(), strike=0.04).to(device)
        derivative.ul().register_buffer("spot", torch.ones(2, 3).to(device))
        result = derivative.payoff()
        expect = torch.full_like(result, -0.04)
        assert_close(result, expect)

        derivative = VarianceSwap(BrownianStock(dt=0.01), strike=0).to(device)
        var = 0.04
        log_return = torch.full((2, 10), sqrt(var * derivative.ul().dt)).to(device)
        log_return[:, 0] = 0.0
        spot = log_return.cumsum(-1).exp()
        derivative.ul().register_buffer("spot", spot)
        result = derivative.payoff()
        expect = torch.full_like(result, var)
        assert_close(result, expect)

        derivative = VarianceSwap(HestonStock())
        derivative.ul().register_buffer("spot", torch.ones(2, 3))
        derivative.ul().register_buffer("variance", torch.ones(2, 3))
        result = derivative.payoff()
        expect = torch.full_like(result, -0.04)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_gpu(self):
        self.test_payoff(device="cuda")

    def test_repr(self):
        derivative = VarianceSwap(BrownianStock())

        result = repr(derivative)
        expect = """\
VarianceSwap(
  strike=0.0400, maturity=0.0800
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert result == expect

    def test_init_dtype_deprecated(self):
        with pytest.raises(DeprecationWarning):
            _ = cls(BrownianStock(), dtype=torch.float64)
