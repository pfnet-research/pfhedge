import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import LookbackOption


class TestLookbackOption:
    """
    pfhedge.instruments.LookbackOption
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self):
        derivative = LookbackOption(BrownianStock(), strike=3.0)
        derivative.underlier.spot = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 2.0], [1.5, 4.0, 1.0]]
        ).T
        # max [2.0, 4.0, 3.0]
        result = derivative.payoff()
        expect = torch.tensor([0.0, 1.0, 0.0])
        assert_close(result, expect)

    def test_payoff_put(self):
        derivative = LookbackOption(BrownianStock(), strike=3.0, call=False)
        derivative.underlier.spot = torch.tensor(
            [[3.0, 6.0, 3.0], [2.0, 5.0, 4.0], [2.5, 4.0, 5.0]]
        ).T
        # min [2.0, 4.0, 3.0]
        result = derivative.payoff()
        expect = torch.tensor([1.0, 0.0, 0.0])
        assert_close(result, expect)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = LookbackOption(BrownianStock(dtype=dtype))
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        derivative = LookbackOption(BrownianStock()).to(dtype=dtype)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    def test_repr(self):
        derivative = LookbackOption(BrownianStock(), maturity=1.0)
        expect = """\
LookbackOption(
  strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect
