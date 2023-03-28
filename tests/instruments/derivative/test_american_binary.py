import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import AmericanBinaryOption
from pfhedge.instruments import BrownianStock

cls = AmericanBinaryOption


class TestAmericanBinaryOption:
    """
    pfhedge.instruments.AmericanBinaryOption
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self, device: str = "cpu"):
        derivative = AmericanBinaryOption(BrownianStock(), strike=2.0).to(device)
        derivative.underlier.register_buffer(
            "spot",
            torch.tensor(
                [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 2.0], [1.9, 2.0, 2.1, 1.0]]
            )
            .to(device)
            .T,
        )
        result = derivative.payoff()
        expect = torch.tensor([0.0, 1.0, 1.0, 1.0]).to(device)
        assert_close(result, expect)

        derivative = AmericanBinaryOption(BrownianStock(), strike=1.0, call=False)
        derivative.underlier.register_buffer(
            "spot",
            torch.tensor(
                [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 1.0], [1.1, 1.0, 0.9, 2.0]]
            )
            .to(device)
            .T,
        )
        result = derivative.payoff()
        expect = torch.tensor([0.0, 1.0, 1.0, 1.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_gpu(self):
        self.test_payoff(device="cuda")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = AmericanBinaryOption(BrownianStock(dtype=dtype, device=device))
        assert derivative.dtype == dtype
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        derivative = AmericanBinaryOption(BrownianStock()).to(
            dtype=dtype, device=device
        )
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    @pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
    def test_device(self, device):
        derivative = AmericanBinaryOption(BrownianStock(device=device))
        assert derivative.device == torch.device(device)

    def test_repr(self):
        derivative = AmericanBinaryOption(BrownianStock(), maturity=1.0)
        expect = """\
AmericanBinaryOption(
  strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

        derivative = AmericanBinaryOption(BrownianStock(), call=False, maturity=1.0)
        expect = """\
AmericanBinaryOption(
  call=False, strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

    def test_init_dtype_deprecated(self):
        with pytest.raises(DeprecationWarning):
            _ = cls(BrownianStock(), dtype=torch.float64)
