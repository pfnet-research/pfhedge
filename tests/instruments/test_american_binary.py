import pytest
import torch

from pfhedge.instruments import AmericanBinaryOption
from pfhedge.instruments import BrownianStock


class TestAmericanBinaryOption:
    """
    pfhedge.instruments.AmericanBinaryOption
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self):
        liability = AmericanBinaryOption(BrownianStock(), strike=2.0)
        liability.underlier.prices = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 2.0], [1.9, 2.0, 2.1, 1.0]]
        ).T
        result = liability.payoff()
        expect = torch.tensor([0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(result, expect)

        liability = AmericanBinaryOption(BrownianStock(), strike=1.0, call=False)
        liability.underlier.prices = torch.tensor(
            [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 1.0], [1.1, 1.0, 0.9, 2.0]]
        ).T
        result = liability.payoff()
        expect = torch.tensor([0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(result, expect)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = AmericanBinaryOption(BrownianStock(dtype=dtype))
        assert liability.dtype == dtype
        liability.simulate()
        assert liability.payoff().dtype == dtype

        liability = AmericanBinaryOption(BrownianStock()).to(dtype=dtype)
        liability.simulate()
        assert liability.payoff().dtype == dtype

    @pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
    def test_device(self, device):
        liability = AmericanBinaryOption(BrownianStock(device=device))
        assert liability.device == torch.device(device)

    def test_repr(self):
        liability = AmericanBinaryOption(BrownianStock(), maturity=1.0)
        expect = (
            "AmericanBinaryOption(BrownianStock(...), strike=1.0, maturity=1.00e+00)"
        )
        assert repr(liability) == expect
        liability = AmericanBinaryOption(BrownianStock(), maturity=1.0, call=False)
        expect = "AmericanBinaryOption(BrownianStock(...), call=False, strike=1.0, maturity=1.00e+00)"
        assert repr(liability) == expect
        liability = AmericanBinaryOption(BrownianStock(), maturity=1.0, strike=2.0)
        expect = (
            "AmericanBinaryOption(BrownianStock(...), strike=2.0, maturity=1.00e+00)"
        )
        assert repr(liability) == expect
