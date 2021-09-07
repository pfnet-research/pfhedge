import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import LookbackOption

cls = LookbackOption


class TestLookbackOption:
    """
    pfhedge.instruments.LookbackOption
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self):
        derivative = LookbackOption(BrownianStock(), strike=3.0)
        derivative.ul().register_buffer(
            "spot", torch.tensor([[1.0, 2.0, 1.5], [2.0, 3.0, 4.0], [3.0, 2.0, 1.0]])
        )
        # max [2.0, 4.0, 3.0]
        result = derivative.payoff()
        expect = torch.tensor([0.0, 1.0, 0.0])
        assert_close(result, expect)

    def test_payoff_put(self):
        derivative = LookbackOption(BrownianStock(), strike=3.0, call=False)
        derivative.ul().register_buffer(
            "spot", torch.tensor([[3.0, 2.0, 2.5], [6.0, 5.0, 4.0], [3.0, 4.0, 5.0]])
        )
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

        derivative = LookbackOption(BrownianStock(), call=False, maturity=1.0)
        expect = """\
LookbackOption(
  call=False, strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

    def test_init_dtype_deprecated(self):
        with pytest.raises(DeprecationWarning):
            _ = cls(BrownianStock(), dtype=torch.float64)
