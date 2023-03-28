import pytest
import torch
from torch import Tensor
from torch.testing import assert_close

from pfhedge.instruments import BaseDerivative
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption

cls = EuropeanOption


class TestEuropeanOption:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), strike=2.0).to(device)
        spot = torch.tensor(
            [[1.0, 1.0, 1.9], [1.0, 1.0, 2.0], [1.0, 1.0, 2.1], [1.0, 1.0, 3.0]]
        ).to(device)
        derivative.underlier.register_buffer("spot", spot)
        result = derivative.payoff()
        expect = torch.tensor([0.0, 0.0, 0.1, 1.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_gpu(self):
        self.test_payoff(device="cuda")

    @pytest.mark.parametrize("volatility", [0.20, 0.10])
    @pytest.mark.parametrize("strike", [1.0, 0.5, 2.0])
    @pytest.mark.parametrize("maturity", [0.1, 1.0])
    @pytest.mark.parametrize("n_paths", [100])
    @pytest.mark.parametrize("init_spot", [1.0, 1.1, 0.9])
    def test_put_call_parity(
        self,
        volatility,
        strike,
        maturity,
        n_paths,
        init_spot,
        device: str = "cpu",
    ):
        stock = BrownianStock(volatility).to(device)
        co = EuropeanOption(stock, strike=strike, maturity=maturity, call=True).to(
            device
        )
        po = EuropeanOption(stock, strike=strike, maturity=maturity, call=False).to(
            device
        )
        co.simulate(n_paths=n_paths, init_state=(init_spot,))
        po.simulate(n_paths=n_paths, init_state=(init_spot,))

        s = stock.spot[..., -1]
        c = co.payoff()
        p = po.payoff()

        assert ((c - p) == s - strike).all()

    @pytest.mark.gpu
    @pytest.mark.parametrize("volatility", [0.20, 0.10])
    @pytest.mark.parametrize("strike", [1.0, 0.5, 2.0])
    @pytest.mark.parametrize("maturity", [0.1, 1.0])
    @pytest.mark.parametrize("n_paths", [100])
    @pytest.mark.parametrize("init_spot", [1.0, 1.1, 0.9])
    def test_put_call_parity_gpu(
        self, volatility, strike, maturity, n_paths, init_spot
    ):
        self.test_put_call_parity(
            volatility, strike, maturity, n_paths, init_spot, device="cuda"
        )

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test_moneyness(self, strike, device: str = "cpu"):
        stock = BrownianStock().to(device)
        derivative = EuropeanOption(stock, strike=strike).to(device)
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test_moneyness_gpu(self, strike):
        self.test_moneyness(strike, device="cuda")

    def test_max_log_moneyness(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.simulate()

        result = derivative.max_log_moneyness(10)
        expect = derivative.max_moneyness(10).log()
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_max_log_moneyness_gpu(self):
        self.test_max_log_moneyness(device="cuda")

    def test_time_to_maturity(self, device: str = "cpu"):
        stock = BrownianStock(dt=0.1).to(device)
        derivative = EuropeanOption(stock, maturity=0.2).to(device)
        derivative.simulate(n_paths=2)

        result = derivative.time_to_maturity()
        expect = torch.tensor([[0.2, 0.1, 0.0], [0.2, 0.1, 0.0]]).to(device)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(0)
        expect = torch.full((2, 1), 0.2).to(device)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(1)
        expect = torch.full((2, 1), 0.1).to(device)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(-1)
        expect = torch.full((2, 1), 0.0).to(device)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    def test_time_to_maturity_gpu(self):
        self.test_time_to_maturity(device="cuda")

    def test_time_to_maturity_2(self, device: str = "cpu"):
        stock = BrownianStock(dt=0.1).to(device)
        derivative = EuropeanOption(stock, maturity=0.25).to(device)
        derivative.simulate(n_paths=2)

        result = derivative.time_to_maturity()
        expect = torch.tensor([[0.3, 0.2, 0.1, 0.0], [0.3, 0.2, 0.1, 0.0]]).to(device)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(0)
        expect = torch.full((2, 1), 0.3).to(device)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(1)
        expect = torch.full((2, 1), 0.2).to(device)
        assert_close(result, expect, check_stride=False)

        result = derivative.time_to_maturity(-1)
        expect = torch.full((2, 1), 0.0).to(device)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    def test_time_to_maturity_2_gpu(self):
        self.test_time_to_maturity_2(device="cuda")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_init_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(dtype=dtype, device=device))
        assert derivative.dtype == dtype

        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_init_dtype_gpu(self, dtype):
        self.test_init_dtype(dtype, device="cuda")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_to_dtype(self, dtype, device: str = "cpu"):
        # to(dtype)
        derivative = EuropeanOption(BrownianStock()).to(dtype).to(device)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        # to(instrument)
        instrument = BrownianStock(dtype=dtype).to(device)
        derivative = EuropeanOption(BrownianStock()).to(instrument).to(device)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        instrument = EuropeanOption(BrownianStock(dtype=dtype)).to(device)
        instrument = BrownianStock(dtype=dtype).to(device)
        derivative = EuropeanOption(BrownianStock()).to(instrument).to(device)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_to_dtype_gpu(self, dtype):
        self.test_to_dtype(dtype, device="cuda")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
    def test_init_device(self, device):
        # init
        derivative = EuropeanOption(BrownianStock(device=device))
        assert derivative.device == torch.device(device)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
    def test_to_device(self, device):
        # to(device)
        s = EuropeanOption(BrownianStock()).to(device)
        assert s.device == torch.device(device)

        # to(instrument)
        instrument = BrownianStock(device=device)
        s = EuropeanOption(BrownianStock()).to(instrument)
        assert s.device == torch.device(device)

        instrument = EuropeanOption(BrownianStock(device=device))
        s = EuropeanOption(BrownianStock()).to(instrument)
        assert s.device == torch.device(device)

    def test_repr(self):
        derivative = EuropeanOption(BrownianStock(), maturity=1.0)
        expect = """\
EuropeanOption(
  strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

        derivative = cls(BrownianStock(), maturity=1.0)
        derivative.add_clause("knockout", lambda derivative, payoff: payoff)
        expect = """\
EuropeanOption(
  strike=1., maturity=1.
  clauses=['knockout']
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

    def test_spot_not_listed(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        with pytest.raises(ValueError):
            _ = derivative.spot
        spot = torch.arange(1.0, 7.0).to(device).reshape(2, 3)
        # tensor([[1., 2., 3.],
        #         [4., 5., 6.]])
        derivative.list(lambda _: spot)
        assert_close(derivative.spot, spot)
        derivative.delist()
        with pytest.raises(ValueError):
            _ = derivative.spot

    @pytest.mark.gpu
    def test_spot_not_listed_gpu(self):
        self.test_spot_not_listed(device="cuda")

    def test_us_listed(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(device=device))
        assert not derivative.is_listed
        derivative.list(pricer=lambda x: x)
        assert derivative.is_listed

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_us_listed_gpu(self, dtype):
        self.test_us_listed(device="cuda")

    def test_init_dtype_deprecated(self):
        with pytest.raises(DeprecationWarning):
            _ = EuropeanOption(BrownianStock(), dtype=torch.float64)

    def test_clause(self, device: str = "cpu"):
        torch.manual_seed(42)

        derivative = cls(BrownianStock()).to(device)
        derivative.simulate()
        strike = derivative.ul().spot.max(-1).values.mean()

        def knockout(derivative: BaseDerivative, payoff: Tensor) -> Tensor:
            max = derivative.ul().spot.max(-1).values
            return payoff.where(max >= strike, torch.zeros_like(max))

        derivative.add_clause("knockout", knockout)

        max = derivative.ul().spot.max(-1).values
        result = derivative.payoff()
        expect = derivative.payoff_fn().where(max >= strike, torch.zeros_like(max))
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_clause_gpu(self):
        self.test_clause(device="cuda")

    def test_add_clause_error(self, device: str = "cpu"):
        derivative = cls(BrownianStock()).to(device)

        def knockout(derivative: BaseDerivative, payoff: Tensor) -> Tensor:
            max = derivative.ul().spot.max(-1).values
            return payoff.where(max >= 1.1, torch.zeros_like(max))

        with pytest.raises(TypeError):
            derivative.add_clause(0, knockout)
        with pytest.raises(KeyError):
            derivative.add_clause("payoff", knockout)
        with pytest.raises(KeyError):
            derivative.add_clause("a.b", knockout)
        with pytest.raises(KeyError):
            derivative.add_clause("", knockout)

    @pytest.mark.gpu
    def test_add_clause_error_gpu(self):
        self.test_add_clause_error(device="cuda")
