import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanForwardStartOption
from pfhedge.instruments import EuropeanOption

cls = EuropeanForwardStartOption


class TestEuropeanOption:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff_start0(self, device: str = "cpu"):
        stock = BrownianStock().to(device)
        derivative = EuropeanForwardStartOption(stock, start=0.0).to(device)
        european = EuropeanOption(stock).to(device)
        derivative.simulate(n_paths=10)
        assert_close(derivative.payoff(), european.payoff())

    @pytest.mark.gpu
    def test_payoff_start0_gpu(self):
        self.test_payoff_start0(device="cuda")

    def test_payoff_start_end(self, device: str = "cpu"):
        stock = BrownianStock().to(device)
        dt = stock.dt
        spot = torch.tensor(
            [[1.0, 2.0, 3.0, 6.0], [1.0, 3.0, 2.0, 6.0], [1.0, 7.0, 1.0, 6.0]]
        ).to(device)
        derivative = EuropeanForwardStartOption(
            stock, maturity=3 * dt, start=1 * dt
        ).to(device)
        derivative.underlier.register_buffer("spot", spot)
        assert_close(derivative.payoff(), torch.tensor([2.0, 1.0, 0.0]).to(device))

        derivative = EuropeanForwardStartOption(
            stock, maturity=3 * dt, start=2 * dt
        ).to(device)
        derivative.underlier.register_buffer("spot", spot)
        assert_close(derivative.payoff(), torch.tensor([1.0, 2.0, 5.0]).to(device))

        # start is rounded off
        derivative = EuropeanForwardStartOption(
            stock, maturity=3 * dt, start=2.9 * dt
        ).to(device)
        derivative.underlier.register_buffer("spot", spot)
        assert_close(derivative.payoff(), torch.tensor([1.0, 2.0, 5.0]).to(device))

    @pytest.mark.gpu
    def test_payoff_start_end_gpu(self):
        self.test_payoff_start_end(device="cuda")

    def test_repr(self):
        derivative = EuropeanForwardStartOption(BrownianStock(), maturity=1.0)
        expect = """\
EuropeanForwardStartOption(
  strike=1., maturity=1., start=0.0400
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect
