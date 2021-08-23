import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanBinaryOption


class TestEuropeanBinaryOption:
    """
    pfhedge.instruments.EuropeanBinaryOption
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self):
        derivative = EuropeanBinaryOption(BrownianStock(), strike=2.0)
        derivative.underlier.spot = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [3.0, 1.0, 1.0, 1.0], [1.9, 2.0, 2.1, 3.0]]
        ).T
        result = derivative.payoff()
        expect = torch.tensor([0.0, 1.0, 1.0, 1.0])
        assert_close(result, expect)

    @pytest.mark.parametrize("volatility", [0.20, 0.10])
    @pytest.mark.parametrize("strike", [1.0, 0.5, 2.0])
    @pytest.mark.parametrize("maturity", [0.1, 1.0])
    @pytest.mark.parametrize("n_paths", [100])
    @pytest.mark.parametrize("init_spot", [1.0, 1.1, 0.9])
    def test_parity(self, volatility, strike, maturity, n_paths, init_spot):
        """
        Test put-call parity.
        """
        stock = BrownianStock(volatility)
        co = EuropeanBinaryOption(stock, strike=strike, maturity=maturity, call=True)
        po = EuropeanBinaryOption(stock, strike=strike, maturity=maturity, call=False)
        co.simulate(n_paths=n_paths, init_state=(init_spot,))
        po.simulate(n_paths=n_paths, init_state=(init_spot,))

        c = co.payoff()
        p = po.payoff()

        assert (c + p == 1.0).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanBinaryOption(BrownianStock(dtype=dtype))
        assert derivative.dtype == dtype
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        derivative = EuropeanBinaryOption(BrownianStock()).to(dtype=dtype)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    @pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
    def test_device(self, device):
        derivative = EuropeanBinaryOption(BrownianStock(device=device))
        assert derivative.device == torch.device(device)

    def test_repr(self):
        derivative = EuropeanBinaryOption(BrownianStock(), maturity=1.0)
        expect = """\
EuropeanBinaryOption(
  strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

        derivative = EuropeanBinaryOption(BrownianStock(), call=False, maturity=1.0)
        expect = """\
EuropeanBinaryOption(
  call=False, strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect
