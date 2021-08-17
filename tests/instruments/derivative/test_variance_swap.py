from math import sqrt

import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import HestonStock
from pfhedge.instruments import VarianceSwap


class TestVarianceSwap:
    def test_repr(self):
        derivative = VarianceSwap(BrownianStock())

        result = repr(derivative)
        expect = "VarianceSwap(BrownianStock(...), strike=4.00e-02, maturity=8.00e-02)"
        assert result == expect

    def test_payoff(self):
        derivative = VarianceSwap(BrownianStock(), strike=0.04)
        derivative.ul().register_buffer("spot", torch.ones(2, 3))
        result = derivative.payoff()
        expect = torch.full_like(result, -0.04)
        assert_close(result, expect)

        derivative = VarianceSwap(BrownianStock(dt=0.01), strike=0)
        var = 0.04
        log_return = torch.full((2, 10), sqrt(var * derivative.ul().dt))
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
