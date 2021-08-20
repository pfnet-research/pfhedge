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
        derivative.underlier.register_buffer("spot", torch.tensor(
            [[1.0, 1.0, 1.9], [1.0, 1.0, 2.0], [1.0, 1.0, 2.1], [1.0, 1.0, 3.0]]
        ))
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

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test_moneyness(self, strike):
        stock = BrownianStock()
        derivative = EuropeanOption(stock, strike=strike)
        derivative.simulate()

        result = derivative.moneyness()
        expect = stock.spot / strike
        assert_close(result, expect)

        result = derivative.moneyness(0)
        expect = stock.spot[:, [0]] / strike
        assert_close(result, expect)

        result = derivative.log_moneyness()
        expect = (stock.spot / strike).log()
        assert_close(result, expect)

        result = derivative.log_moneyness(0)
        expect = (stock.spot[:, [0]] / strike).log()
        assert_close(result, expect)

    def test_time_to_maturity(self):
        stock = BrownianStock(dt=0.1)
        derivative = EuropeanOption(stock, maturity=0.2)
        derivative.simulate(n_paths=2)

        result = derivative.time_to_maturity()
        expect = torch.tensor([[0.2, 0.1, 0.0], [0.2, 0.1, 0.0]])
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(0)
        expect = torch.full((2, 1), 0.2)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(1)
        expect = torch.full((2, 1), 0.1)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(-1)
        expect = torch.full((2, 1), 0.0)
        assert_close(result, expect, check_stride=False)

    def test_time_to_maturity_2(self):
        stock = BrownianStock(dt=0.1)
        derivative = EuropeanOption(stock, maturity=0.25)
        derivative.simulate(n_paths=2)

        result = derivative.time_to_maturity()
        expect = torch.tensor([[0.3, 0.2, 0.1, 0.0], [0.3, 0.2, 0.1, 0.0]])
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(0)
        expect = torch.full((2, 1), 0.3)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(1)
        expect = torch.full((2, 1), 0.2)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(-1)
        expect = torch.full((2, 1), 0.0)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock(dtype=dtype))
        assert derivative.dtype == dtype

        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        derivative = EuropeanOption(BrownianStock()).to(dtype=dtype)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    @pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
    def test_device(self, device):
        derivative = EuropeanOption(BrownianStock(device=device))
        assert derivative.device == torch.device(device)

    def test_repr(self):
        derivative = EuropeanOption(BrownianStock(), maturity=1.0)
        expect = """\
EuropeanOption(
  strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

        derivative = EuropeanOption(BrownianStock(), maturity=1.0, call=False)
        expect = """\
EuropeanOption(
  call=False, strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

        derivative = EuropeanOption(BrownianStock(), maturity=1.0, strike=2.0)
        expect = """\
EuropeanOption(
  strike=2., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

        derivative = EuropeanOption(BrownianStock(), maturity=1.0)
        derivative.to(dtype=torch.float64, device="cuda:0")
        expect = """\
EuropeanOption(
  strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040, dtype=torch.float64, device='cuda:0')
)"""
        assert repr(derivative) == expect

    def test_spot_not_listed(self):
        derivative = EuropeanOption(BrownianStock())
        with pytest.raises(ValueError):
            _ = derivative.spot
