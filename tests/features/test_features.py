import pytest
import torch
from torch.nn import Linear
from torch.testing import assert_close

from pfhedge.features import Barrier
from pfhedge.features import Empty
from pfhedge.features import ExpiryTime
from pfhedge.features import FeatureList
from pfhedge.features import LogMoneyness
from pfhedge.features import MaxLogMoneyness
from pfhedge.features import MaxMoneyness
from pfhedge.features import ModuleOutput
from pfhedge.features import Moneyness
from pfhedge.features import PrevHedge
from pfhedge.features import Spot
from pfhedge.features import TimeToMaturity
from pfhedge.features import UnderlierSpot
from pfhedge.features import Variance
from pfhedge.features import Volatility
from pfhedge.features import Zeros
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.instruments import HestonStock
from pfhedge.nn import Hedger
from pfhedge.nn import Naked


class _TestFeature:
    def assert_same_dtype(
        self,
        feature,
        derivative,
        dtype,
        device: str = "cpu",
    ):
        derivative.to(dtype=dtype, device=device).simulate()
        f = feature.of(derivative)
        assert f.get(0).dtype == dtype


class TestMoneyness(_TestFeature):
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test_value(self, strike, log, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), strike=strike).to(device)
        spot = torch.arange(1.0, 7.0).to(device).reshape(2, 3)
        # tensor([[1., 2., 3.],
        #         [4., 5., 6.]])
        derivative.underlier.register_buffer("spot", spot)
        f = Moneyness(log=log).of(derivative)

        result = f.get(0)
        expect = torch.tensor([[1.0], [4.0]]).to(device) / strike
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = torch.tensor([[2.0], [5.0]]).to(device) / strike
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(2)
        expect = torch.tensor([[3.0], [6.0]]).to(device) / strike
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get()
        expect = spot / strike
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test_value_gpu(self, strike, log):
        self.test_value(strike, log, device="cuda")

    def test_str(self):
        assert str(Moneyness()) == "moneyness"
        assert str(Moneyness(log=True)) == "log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(Moneyness(), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = Moneyness().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")

    # def test_getitem_deprecation_warning(
    #     self, device: str = "cpu"
    # ):
    #     derivative = EuropeanOption(BrownianStock()).to(device)
    #     spot = torch.zeros(2, 3).to(device)
    #     derivative.underlier.register_buffer("spot", spot)
    #     f = Moneyness().of(derivative)
    #
    #     with pytest.raises(DeprecationWarning):
    #         _ = f[0]
    #
    # @pytest.mark.gpu
    # def test_getitem_deprecation_warning_gpu(self):
    #     self.test_getitem_deprecation_warning(device="cuda")

    @pytest.mark.filterwarnings("ignore")
    def test_getitem_get(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        spot = torch.zeros(2, 3).to(device)
        derivative.underlier.register_buffer("spot", spot)
        f = Moneyness().of(derivative)
        assert_close(f.get(0), f[0])

    @pytest.mark.gpu
    @pytest.mark.filterwarnings("ignore")
    def test_getitem_get_gpu(self):
        self.test_getitem_get(device="cuda")


class TestLogMoneyness(_TestFeature):
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test_value(self, strike, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), strike=strike).to(device)
        derivative.ul().register_buffer(
            "spot", torch.arange(1.0, 7.0).to(device).reshape(2, 3)
        )
        # tensor([[1., 2., 3.],
        #         [4., 5., 6.]])
        f = LogMoneyness().of(derivative)

        result = f.get(0)
        expect = (torch.tensor([[1.0], [4.0]]).to(device) / strike).log()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = (torch.tensor([[2.0], [5.0]]).to(device) / strike).log()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(2)
        expect = (torch.tensor([[3.0], [6.0]]).to(device) / strike).log()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test_value_gpu(self, strike):
        self.test_value(strike, device="cuda")

    def test_str(self):
        assert str(LogMoneyness()) == "log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(LogMoneyness(), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = LogMoneyness().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestTimeToMaturity(_TestFeature):
    def test(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(dt=0.1), maturity=0.2).to(device)
        derivative.ul().register_buffer("spot", torch.zeros(2, 3).to(device))
        f = TimeToMaturity().of(derivative)

        result = f.get(0)
        expect = torch.full((2, 1), 0.2).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get(1)
        expect = torch.full((2, 1), 0.1).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get(2)
        expect = torch.full((2, 1), 0.0).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get()
        expect = torch.tensor([[0.2, 0.1, 0.0], [0.2, 0.1, 0.0]]).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    def test_gpu(self):
        self.test(device="cuda")

    def test_2(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(dt=0.1), maturity=0.15).to(device)
        derivative.underlier.register_buffer("spot", torch.zeros(2, 3).to(device))
        f = TimeToMaturity().of(derivative)

        result = f.get(0)
        expect = torch.full((2, 1), 0.2).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get(1)
        expect = torch.full((2, 1), 0.1).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get(2)
        expect = torch.full((2, 1), 0.0).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get()
        expect = torch.tensor([[0.2, 0.1, 0.0], [0.2, 0.1, 0.0]]).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    def test_2_gpu(self):
        self.test_2(device="cuda")

    def test_str(self):
        assert str(TimeToMaturity()) == "time_to_maturity"
        assert str(ExpiryTime()) == "expiry_time"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(TimeToMaturity(), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = TimeToMaturity().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestVolatility(_TestFeature):
    @pytest.mark.parametrize("sigma", [0.2, 0.1])
    def test_constant_volatility(self, sigma, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(sigma=sigma)).to(device)
        derivative.underlier.register_buffer("spot", torch.zeros(2, 3).to(device))

        f = Volatility().of(derivative)

        result = f.get(0)
        expect = torch.full((2, 1), sigma).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get(1)
        expect = torch.full((2, 1), sigma).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get(2)
        expect = torch.full((2, 1), sigma).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get()
        expect = torch.full((2, 3), sigma).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    @pytest.mark.parametrize("sigma", [0.2, 0.1])
    def test_constant_volatility_gpu(self, sigma):
        self.test_constant_volatility(sigma, device="cuda")

    def test_stochastic_volatility(self, device: str = "cpu"):
        derivative = EuropeanOption(HestonStock(dt=0.1), maturity=0.2).to(device)
        derivative.simulate(n_paths=2)
        variance = derivative.ul().variance

        f = Volatility().of(derivative)

        result = f.get(0)
        expect = variance[:, [0]].sqrt()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get(1)
        expect = variance[:, [1]].sqrt()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f.get()
        expect = variance.sqrt()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    def test_stochastic_volatility_gpu(self):
        self.test_stochastic_volatility(device="cuda")

    def test_str(self):
        assert str(Volatility()) == "volatility"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(Volatility(), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = Volatility().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestVariance(_TestFeature):
    @pytest.mark.parametrize("sigma", [0.2, 0.1])
    def test_constant_volatility(self, sigma, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(sigma=sigma)).to(device)
        derivative.underlier.register_buffer("spot", torch.zeros(2, 3).to(device))

        f = Variance().of(derivative)

        result = f[0]
        expect = torch.full((2, 1), sigma ** 2).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f[1]
        expect = torch.full((2, 1), sigma ** 2).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f[2]
        expect = torch.full((2, 1), sigma ** 2).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f[None]
        expect = torch.full((2, 3), sigma ** 2).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    @pytest.mark.parametrize("sigma", [0.2, 0.1])
    def test_constant_volatility_gpu(self, sigma):
        self.test_constant_volatility(sigma, device="cuda")

    def test_stochastic_volatility(self, device: str = "cpu"):
        derivative = EuropeanOption(HestonStock(dt=0.1), maturity=0.2).to(device)
        derivative.simulate(n_paths=2)
        variance = derivative.ul().variance

        f = Variance().of(derivative)

        result = f[0]
        expect = variance[:, [0]]
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f[1]
        expect = variance[:, [1]]
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

        result = f[None]
        expect = variance
        expect = expect.unsqueeze(-1)
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    def test_stochastic_volatility_gpu(self):
        self.test_stochastic_volatility(device="cuda")

    def test_str(self):
        assert str(Variance()) == "variance"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(Variance(), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = Variance().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestPrevHedge(_TestFeature):
    @pytest.mark.parametrize("volatility", [0.2, 0.1])
    def test(self, volatility, device: str = "cpu"):
        N, T = 10, 20
        torch.manual_seed(42)
        derivative = EuropeanOption(BrownianStock(volatility)).to(device)
        derivative.ul().register_buffer("spot", torch.randn(N, T).to(device))
        hedger = Hedger(Linear(2, 1), ["empty", "empty"]).to(device)

        f = PrevHedge().of(derivative, hedger)

        with pytest.raises(AttributeError):
            _ = f.get(0)
            # expect = torch.zeros((N, 1, 1))
            # assert_close(result, expect)

        # input = torch.cat([feature[0] for feature in hedger.inputs], dim=-1)
        input = torch.randn(N, 1, 2).to(device)
        expect = hedger(input)
        result = f.get(1)
        assert_close(result, expect)

        # input = torch.cat([feature[1] for feature in hedger.inputs], dim=-1)
        input = torch.randn(N, 1, 2).to(device)
        expect = hedger(input)
        result = f.get(2)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("volatility", [0.2, 0.1])
    def test_gpu(self, volatility):
        self.test(volatility, device="cuda")

    def test_str(self):
        assert str(PrevHedge()) == "prev_hedge"

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = PrevHedge().of(derivative, hedger)
        assert f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")

    def test_error_time_step_is_none(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = PrevHedge().of(derivative, hedger)
        with pytest.raises(ValueError):
            _ = f.get(None)

    @pytest.mark.gpu
    def test_error_time_step_is_none_gpu(self):
        self.test_error_time_step_is_none(device="cuda")


class TestBarrier(_TestFeature):
    def test(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.ul().register_buffer(
            "spot",
            torch.tensor(
                [
                    [1.0, 1.5, 2.0, 3.0],
                    [2.0, 1.0, 1.0, 1.0],
                    [3.0, 4.0, 5.0, 6.0],
                    [1.0, 1.1, 1.2, 1.3],
                ]
            ).to(device),
        )
        f = Barrier(2.0, up=True).of(derivative)

        result = f.get(0)
        expect = torch.tensor([0.0, 1.0, 1.0, 0.0]).to(device)
        expect = expect.unsqueeze(-1).unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = torch.tensor([0.0, 1.0, 1.0, 0.0]).to(device)
        expect = expect.unsqueeze(-1).unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(2)
        expect = torch.tensor([1.0, 1.0, 1.0, 0.0]).to(device)
        expect = expect.unsqueeze(-1).unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(3)
        expect = torch.tensor([1.0, 1.0, 1.0, 0.0]).to(device)
        expect = expect.unsqueeze(-1).unsqueeze(-1)
        assert_close(result, expect)

        result = f.get()
        expect = (
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
            .to(device)
            .unsqueeze(-1)
        )
        assert_close(result, expect)

        derivative = EuropeanOption(BrownianStock())
        derivative.underlier.register_buffer(
            "spot",
            torch.tensor(
                [
                    [3.0, 2.0, 1.5, 1.0],
                    [1.0, 1.0, 1.0, 2.0],
                    [6.0, 5.0, 4.0, 3.0],
                    [1.3, 1.2, 1.1, 1.0],
                ]
            ).to(device),
        )
        f = Barrier(2.0, up=False).of(derivative)

        result = f.get(0)
        expect = torch.tensor([0.0, 1.0, 0.0, 1.0]).to(device).reshape(-1, 1)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = torch.tensor([1.0, 1.0, 0.0, 1.0]).to(device).reshape(-1, 1)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(2)
        expect = torch.tensor([1.0, 1.0, 0.0, 1.0]).to(device).reshape(-1, 1)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(3)
        expect = torch.tensor([1.0, 1.0, 0.0, 1.0]).to(device).reshape(-1, 1)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(None)
        expect = (
            torch.tensor(
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ]
            )
            .to(device)
            .unsqueeze(-1)
        )
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_gpu(self):
        self.test(device="cuda")

    def test_repr(self):
        assert repr(Barrier(1.0, up=True)) == "Barrier(1., up=True)"
        assert repr(Barrier(2.0, up=True)) == "Barrier(2., up=True)"
        assert repr(Barrier(1.0, up=False)) == "Barrier(1., up=False)"
        assert repr(Barrier(2.0, up=False)) == "Barrier(2., up=False)"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(Barrier(1.0), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = Barrier(1.0).of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestZeros(_TestFeature):
    """
    pfhedge.features.Zeros
    """

    def test(self, device: str = "cpu"):
        torch.manual_seed(42)
        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.ul().register_buffer("spot", torch.zeros(2, 3).to(device))

        f = Zeros().of(derivative)

        result = f.get(0)
        expect = torch.zeros((2, 1)).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = torch.zeros((2, 1)).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(2)
        expect = torch.zeros((2, 1)).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(None)
        expect = torch.zeros((2, 3)).to(device)
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_gpu(self):
        self.test(device="cuda")

    def test_str(self):
        assert str(Zeros()) == "zeros"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(Zeros(), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = Zeros().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestEmpty(_TestFeature):
    def test(self, device: str = "cpu"):
        torch.manual_seed(42)
        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.underlier.register_buffer("spot", torch.zeros(2, 3).to(device))

        f = Empty().of(derivative)

        result = f.get(0)
        assert result.size() == torch.Size((2, 1, 1))

        result = f.get(1)
        assert result.size() == torch.Size((2, 1, 1))

        result = f.get(2)
        assert result.size() == torch.Size((2, 1, 1))

        result = f.get(None)
        assert result.size() == torch.Size((2, 3, 1))

    @pytest.mark.gpu
    def test_gpu(self):
        self.test(device="cuda")

    def test_str(self):
        assert str(Empty()) == "empty"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(Empty(), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = Empty().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestMaxMoneyness(_TestFeature):
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test(self, strike, log, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), strike=strike).to(device)
        derivative.ul().register_buffer(
            "spot",
            torch.tensor([[1.0, 2.0, 1.5], [2.0, 3.0, 4.0], [3.0, 2.0, 1.0]]).to(
                device
            ),
        )

        f = MaxMoneyness(log=log).of(derivative)

        result = f.get(0)
        expect = torch.tensor([[1.0], [2.0], [3.0]]).to(device) / strike
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = torch.tensor([[2.0], [3.0], [3.0]]).to(device) / strike
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(2)
        expect = torch.tensor([[2.0], [4.0], [3.0]]).to(device) / strike
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(None)
        expect = (
            torch.tensor([[1.0, 2.0, 2.0], [2.0, 3.0, 4.0], [3.0, 3.0, 3.0]]).to(device)
            / strike
        )
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test_gpu(self, strike, log):
        self.test(strike, log, device="cuda")

    def test_str(self):
        assert str(MaxMoneyness()) == "max_moneyness"
        assert str(MaxMoneyness(log=True)) == "max_log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(MaxMoneyness(), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = MaxMoneyness().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestMaxLogMoneyness(_TestFeature):
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test(self, strike, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), strike=strike).to(device)
        derivative.ul().register_buffer(
            "spot",
            torch.tensor([[1.0, 2.0, 1.5], [2.0, 3.0, 4.0], [3.0, 2.0, 1.0]]).to(
                device
            ),
        )

        f = MaxLogMoneyness().of(derivative)

        result = f.get(0)
        expect = (torch.tensor([[1.0], [2.0], [3.0]]).to(device) / strike).log()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = (torch.tensor([[2.0], [3.0], [3.0]]).to(device) / strike).log()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(2)
        expect = (torch.tensor([[2.0], [4.0], [3.0]]).to(device) / strike).log()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(None)
        expect = (
            torch.tensor([[1.0, 2.0, 2.0], [2.0, 3.0, 4.0], [3.0, 3.0, 3.0]]).to(device)
            / strike
        ).log()
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test_gpu(self, strike):
        self.test(strike, device="cuda")

    def test_str(self):
        assert str(MaxLogMoneyness()) == "max_log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        self.assert_same_dtype(MaxLogMoneyness(), derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = MaxLogMoneyness().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestModuleOutput(_TestFeature):
    def test(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.simulate()

        module = torch.nn.Linear(2, 1).to(device)
        x1, x2 = Moneyness(), TimeToMaturity()
        f = ModuleOutput(module, [x1, x2]).of(derivative)

        result = f.get(0)
        expect = module(
            torch.cat([x1.of(derivative).get(0), x2.of(derivative).get(0)], dim=-1)
        )
        assert_close(result, expect)

        result = f.get(1)
        expect = module(
            torch.cat([x1.of(derivative).get(1), x2.of(derivative).get(1)], dim=-1)
        )
        assert_close(result, expect)

        result = f.get(2)
        expect = module(
            torch.cat([x1.of(derivative).get(2), x2.of(derivative).get(2)], dim=-1)
        )
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_gpu(self):
        self.test(device="cuda")

    def test_repr(self):
        module = torch.nn.Linear(2, 1)
        x1, x2 = Moneyness(), TimeToMaturity()
        f = ModuleOutput(module, [x1, x2])
        expect = (
            "ModuleOutput(\n"
            "  inputs=['moneyness', 'time_to_maturity']\n"
            "  (module): Linear(in_features=2, out_features=1, bias=True)\n"
            ")"
        )
        assert repr(f) == expect

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device, dtype)
        m = Linear(2, 1).to(derivative.device, derivative.dtype)
        f = ModuleOutput(m, [Moneyness(), TimeToMaturity()])
        self.assert_same_dtype(f, derivative, dtype, device)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_gpu(self, dtype):
        self.test_dtype(dtype, device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)

        f = ModuleOutput(Linear(2, 1).to(device), [Moneyness(), TimeToMaturity()])
        f = f.of(derivative, hedger)
        assert not f.is_state_dependent()

        f = ModuleOutput(Linear(2, 1).to(device), [Moneyness(), PrevHedge()])
        f = f.of(derivative, hedger)
        assert f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestFeatureList:
    def test_repr(self):
        derivative = EuropeanOption(BrownianStock())
        derivative.simulate()
        hedger = Hedger(Naked(), inputs=["empty"])
        f = FeatureList(["moneyness", "expiry_time"])
        f = f.of(derivative, hedger)

        assert repr(f) == "['moneyness', 'expiry_time']"

    def test_len(self):
        f = FeatureList(["empty", "empty"])
        assert len(f) == 2

    def test_value(self, device: str = "cpu"):
        torch.manual_seed(42)

        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.simulate()
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)

        f0 = Moneyness().of(derivative, hedger)
        f1 = ExpiryTime().of(derivative, hedger)
        f = FeatureList([f0, f1]).of(derivative, hedger)

        result = f.get(0)
        expect = torch.cat([f0.get(0), f1.get(0)], dim=-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = torch.cat([f0.get(1), f1.get(1)], dim=-1)
        assert_close(result, expect)

        result = f.get(None)
        expect = torch.cat([f0.get(None), f1.get(None)], dim=-1)
        assert_close(result, expect)

        torch.manual_seed(42)

        f0 = Moneyness().of(derivative, hedger)
        f1 = PrevHedge().of(derivative, hedger)
        f = FeatureList([f0, f1]).of(derivative, hedger)

        hedger(torch.ones(derivative.ul().spot.size(0), 1, 2).to(device))
        result = f.get(1)
        expect = torch.cat([f0.get(1), f1.get(1)], dim=-1)
        assert_close(result, expect)

        hedger(torch.ones(derivative.ul().spot.size(0), 1, 2).to(device))
        result = f.get(2)
        expect = torch.cat([f0.get(2), f1.get(2)], dim=-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_value_gpu(self):
        self.test_value(device="cuda")

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.simulate()
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)

        f = FeatureList(["moneyness", "expiry_time"])
        f = f.of(derivative, hedger)

        assert not f.is_state_dependent()

        f = FeatureList(["moneyness", "prev_hedge"])
        f = f.of(derivative, hedger)

        assert f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestUnderlierSpot(_TestFeature):
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test_value(self, strike, log, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), strike=strike).to(device)
        spot = torch.arange(1.0, 7.0).to(device).reshape(2, 3)
        # tensor([[1., 2., 3.],
        #         [4., 5., 6.]])
        derivative.underlier.register_buffer("spot", spot)
        f = UnderlierSpot(log=log).of(derivative)

        result = f.get(0)
        expect = torch.tensor([[1.0], [4.0]]).to(device)
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = torch.tensor([[2.0], [5.0]]).to(device)
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(2)
        expect = torch.tensor([[3.0], [6.0]]).to(device)
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get()
        expect = spot
        expect = expect.log_() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test_value_gpu(self, strike, log):
        self.test_value(strike, log, device="cuda")

    def test_str(self):
        assert str(UnderlierSpot()) == "underlier_spot"

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = UnderlierSpot().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")


class TestSpot(_TestFeature):
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test_value(self, strike, log, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), strike=strike).to(device)
        spot = torch.arange(1.0, 7.0).to(device).reshape(2, 3)
        # tensor([[1., 2., 3.],
        #         [4., 5., 6.]])
        derivative.list(lambda _: spot)
        f = Spot(log=log).of(derivative)

        result = f.get(0)
        expect = torch.tensor([[1.0], [4.0]]).to(device)
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(1)
        expect = torch.tensor([[2.0], [5.0]]).to(device)
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get(2)
        expect = torch.tensor([[3.0], [6.0]]).to(device)
        expect = expect.log() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

        result = f.get()
        expect = spot
        expect = expect.log_() if log else expect
        expect = expect.unsqueeze(-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test_value_gpu(self, strike, log):
        self.test_value(strike, log, device="cuda")

    def test_str(self):
        assert str(Spot()) == "spot"

    def test_is_state_dependent(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Naked(), inputs=["empty"]).to(device)
        f = Spot().of(derivative, hedger)
        assert not f.is_state_dependent()

    @pytest.mark.gpu
    def test_is_state_dependent_gpu(self):
        self.test_is_state_dependent(device="cuda")
