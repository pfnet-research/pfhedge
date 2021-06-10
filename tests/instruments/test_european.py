import pytest
import torch

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption


class TestEuropeanOption:
    """
    pfhedge.instruments.EuropeanOption
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self):
        liability = EuropeanOption(BrownianStock(), strike=2.0)
        liability.underlier.prices = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.9, 2.0, 2.1, 3.0]]
        )
        result = liability.payoff()
        expect = torch.tensor([0.0, 0.0, 0.1, 1.0])
        assert torch.allclose(result, expect)

    @pytest.mark.parametrize("volatility", [0.20, 0.10])
    @pytest.mark.parametrize("strike", [1.0, 0.5, 2.0])
    @pytest.mark.parametrize("maturity", [0.1, 1.0])
    @pytest.mark.parametrize("n_paths", [100])
    @pytest.mark.parametrize("init_price", [1.0, 1.1, 0.9])
    def test_parity(self, volatility, strike, maturity, n_paths, init_price):
        """
        Test put-call parity.
        """
        stock = BrownianStock(volatility)
        co = EuropeanOption(stock, strike=strike, maturity=maturity, call=True)
        po = EuropeanOption(stock, strike=strike, maturity=maturity, call=False)
        co.simulate(n_paths=n_paths, init_price=init_price)
        po.simulate(n_paths=n_paths, init_price=init_price)

        s = stock.prices[-1, :]
        c = co.payoff()
        p = po.payoff()

        assert ((c - p) == s - strike).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock(dtype=dtype))
        assert liability.dtype == dtype
        liability.simulate()
        assert liability.payoff().dtype == dtype

        liability = EuropeanOption(BrownianStock()).to(dtype=dtype)
        liability.simulate()
        assert liability.payoff().dtype == dtype

    @pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
    def test_device(self, device):
        liability = EuropeanOption(BrownianStock(device=device))
        assert liability.device == torch.device(device)

    def test_repr(self):
        liability = EuropeanOption(BrownianStock(), maturity=1.0)
        expect = "EuropeanOption(BrownianStock(...), strike=1.0, maturity=1.00e+00)"
        assert repr(liability) == expect

        liability = EuropeanOption(BrownianStock(), maturity=1.0, call=False)
        expect = "EuropeanOption(BrownianStock(...), call=False, strike=1.0, maturity=1.00e+00)"
        assert repr(liability) == expect

        liability = EuropeanOption(BrownianStock(), maturity=1.0, strike=2.0)
        expect = "EuropeanOption(BrownianStock(...), strike=2.0, maturity=1.00e+00)"
        assert repr(liability) == expect

        liability = EuropeanOption(BrownianStock(), maturity=1.0)
        liability.to(dtype=torch.float64, device="cuda:0")
        expect = (
            "EuropeanOption(BrownianStock(...), strike=1.0, maturity=1.00e+00,"
            " dtype=torch.float64, device='cuda:0')"
        )
        assert repr(liability) == expect
