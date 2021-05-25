import pytest
import torch

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
        liability = LookbackOption(BrownianStock(), strike=3.0)
        liability.underlier.prices = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 2.0], [1.5, 4.0, 1.0]]
        )
        # max [2.0, 4.0, 3.0]
        result = liability.payoff()
        expect = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(result, expect)

    def test_payoff_put(self):
        liability = LookbackOption(BrownianStock(), strike=3.0, call=False)
        liability.underlier.prices = torch.tensor(
            [[3.0, 6.0, 3.0], [2.0, 5.0, 4.0], [2.5, 4.0, 5.0]]
        )
        # min [2.0, 4.0, 3.0]
        result = liability.payoff()
        expect = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(result, expect)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = LookbackOption(BrownianStock(dtype=dtype))
        liability.simulate()
        assert liability.payoff().dtype == dtype

        liability = LookbackOption(BrownianStock()).to(dtype=dtype)
        liability.simulate()
        assert liability.payoff().dtype == dtype

    def test_repr(self):
        liability = LookbackOption(BrownianStock(), maturity=1.0)
        expect = "LookbackOption(BrownianStock(...), maturity=1.00e+00)"
        assert repr(liability) == expect
        liability = LookbackOption(BrownianStock(), maturity=1.0, call=False)
        expect = "LookbackOption(BrownianStock(...), call=False, maturity=1.00e+00)"
        assert repr(liability) == expect
        liability = LookbackOption(BrownianStock(), maturity=1.0, strike=2.0)
        expect = "LookbackOption(BrownianStock(...), strike=2.0, maturity=1.00e+00)"
        assert repr(liability) == expect
