import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption


class TestEuropeanOption:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self):
        derivative = EuropeanOption(BrownianStock(), strike=2.0)
        derivative.underlier.spot = torch.tensor(
            [[1.0, 1.0, 1.9], [1.0, 1.0, 2.0], [1.0, 1.0, 2.1], [1.0, 1.0, 3.0]]
        )
        result = derivative.payoff()
        expect = torch.tensor([0.0, 0.0, 0.1, 1.0])
        assert_close(result, expect)

    @pytest.mark.parametrize("volatility", [0.20, 0.10])
    @pytest.mark.parametrize("strike", [1.0, 0.5, 2.0])
    @pytest.mark.parametrize("maturity", [0.1, 1.0])
    @pytest.mark.parametrize("n_paths", [100])
    @pytest.mark.parametrize("init_spot", [1.0, 1.1, 0.9])
    def test_put_call_parity(self, volatility, strike, maturity, n_paths, init_spot):
        stock = BrownianStock(volatility)
        co = EuropeanOption(stock, strike=strike, maturity=maturity, call=True)
        po = EuropeanOption(stock, strike=strike, maturity=maturity, call=False)
        co.simulate(n_paths=n_paths, init_state=(init_spot,))
        po.simulate(n_paths=n_paths, init_state=(init_spot,))

        s = stock.spot[..., -1]
        c = co.payoff()
        p = po.payoff()

        assert ((c - p) == s - strike).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock(dtype=dtype))
        assert derivative.dtype == dtype

        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        derivative = EuropeanOption(BrownianStock()).to(dtype=dtype)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    @pytest.mark.parametrize("device", ["cpu","cuda:0", "cuda:1"])
    def test_device(self, device):
        derivative = EuropeanOption(BrownianStock(device=device))
        assert derivative.device == torch.device(device)

    def test_repr(self):
        derivative = EuropeanOption(BrownianStock(), maturity=1.0)
        expect = "EuropeanOption(BrownianStock(...), strike=1.0, maturity=1.00e+00)"
        assert repr(derivative) == expect

        derivative = EuropeanOption(BrownianStock(), maturity=1.0, call=False)
        expect = "EuropeanOption(BrownianStock(...), call=False, strike=1.0, maturity=1.00e+00)"
        assert repr(derivative) == expect

        derivative = EuropeanOption(BrownianStock(), maturity=1.0, strike=2.0)
        expect = "EuropeanOption(BrownianStock(...), strike=2.0, maturity=1.00e+00)"
        assert repr(derivative) == expect

        derivative = EuropeanOption(BrownianStock(), maturity=1.0)
        derivative.to(dtype=torch.float64, device="cuda:0")
        expect = (
            "EuropeanOption(BrownianStock(...), strike=1.0, maturity=1.00e+00,"
            " dtype=torch.float64, device='cuda:0')"
        )
        assert repr(derivative) == expect
