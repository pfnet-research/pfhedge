import pytest
import torch
from torch import Tensor
from torch.testing import assert_close

from pfhedge.instruments import BaseDerivative
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanForwardStartOption
from pfhedge.instruments import EuropeanOption

cls = EuropeanForwardStartOption


class TestEuropeanOption:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff_start0(self):
        stock = BrownianStock()
        derivative = EuropeanForwardStartOption(stock, start=0.0)
        european = EuropeanOption(stock)
        derivative.simulate(n_paths=10)
        assert_close(derivative.payoff(), european.payoff())

    def test_payoff_start_end(self):
        stock = BrownianStock()
        dt = stock.dt
        spot = torch.tensor(
            [[1.0, 2.0, 3.0, 6.0], [1.0, 3.0, 2.0, 6.0], [1.0, 7.0, 1.0, 6.0]]
        )
        derivative = EuropeanForwardStartOption(stock, maturity=3 * dt, start=1 * dt)
        derivative.underlier.register_buffer("spot", spot)
        assert_close(derivative.payoff(), torch.tensor([2.0, 1.0, 0.0]))

        derivative = EuropeanForwardStartOption(stock, maturity=3 * dt, start=2 * dt)
        derivative.underlier.register_buffer("spot", spot)
        assert_close(derivative.payoff(), torch.tensor([1.0, 2.0, 5.0]))

        # start is rounded off
        derivative = EuropeanForwardStartOption(stock, maturity=3 * dt, start=2.9 * dt)
        derivative.underlier.register_buffer("spot", spot)
        assert_close(derivative.payoff(), torch.tensor([1.0, 2.0, 5.0]))

    def test_repr(self):
        derivative = EuropeanForwardStartOption(BrownianStock(), maturity=1.0)
        expect = """\
EuropeanForwardStartOption(
  strike=1., maturity=1., start=0.0400
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect
