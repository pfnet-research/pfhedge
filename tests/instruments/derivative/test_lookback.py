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

    def test_payoff(self, device: str = "cpu"):
        derivative = LookbackOption(BrownianStock(), strike=3.0).to(device)
        derivative.ul().register_buffer(
            "spot",
            torch.tensor([[1.0, 2.0, 1.5], [2.0, 3.0, 4.0], [3.0, 2.0, 1.0]]).to(
                device
            ),
        )
        # max [2.0, 4.0, 3.0]
        result = derivative.payoff()
        expect = torch.tensor([0.0, 1.0, 0.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_gpu(self):
        self.test_payoff(device="cuda")

    def test_payoff_put(self, device: str = "cpu"):
        derivative = LookbackOption(BrownianStock(), strike=3.0, call=False).to(device)
        derivative.ul().register_buffer(
            "spot",
            torch.tensor([[3.0, 2.0, 2.5], [6.0, 5.0, 4.0], [3.0, 4.0, 5.0]]).to(
                device
            ),
        )
        # min [2.0, 4.0, 3.0]
        result = derivative.payoff()
        expect = torch.tensor([1.0, 0.0, 0.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_put_gpu(self):
        self.test_payoff_put(device="cuda")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = LookbackOption(BrownianStock(dtype=dtype, device=device))
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        derivative = LookbackOption(BrownianStock()).to(dtype=dtype, device=device)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

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
