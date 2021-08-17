import pytest
import torch
from torch.nn import Linear
from torch.testing import assert_close

from pfhedge.features import Barrier
from pfhedge.features import Empty
from pfhedge.features import ExpiryTime
from pfhedge.features import LogMoneyness
from pfhedge.features import MaxLogMoneyness
from pfhedge.features import MaxMoneyness
from pfhedge.features import ModuleOutput
from pfhedge.features import Moneyness
from pfhedge.features import PrevHedge
from pfhedge.features import Volatility
from pfhedge.features import Zeros
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.instruments import HestonStock
from pfhedge.nn import Hedger


class _TestFeature:
    def assert_same_dtype(self, feature, derivative, dtype):
        derivative.to(dtype).simulate()
        assert feature.of(derivative)[0].dtype == dtype


class TestMoneyness(_TestFeature):
    """
    pfhedge.features.Moneyness
    """

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test(self, strike, log):
        derivative = EuropeanOption(BrownianStock(), strike=strike)
        derivative.underlier.spot = torch.arange(1.0, 7.0).reshape(2, 3)
        # tensor([[1., 2., 3.],
        #         [4., 5., 6.]])
        f = Moneyness(log=log).of(derivative)

        result = f[0]
        expect = torch.tensor([[1.0], [4.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert_close(result, expect)
        result = f[1]
        expect = torch.tensor([[2.0], [5.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert_close(result, expect)
        result = f[2]
        expect = torch.tensor([[3.0], [6.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert_close(result, expect)

    def test_str(self):
        assert str(Moneyness()) == "moneyness"
        assert str(Moneyness(log=True)) == "log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock())
        self.assert_same_dtype(Moneyness(), derivative, dtype)


class TestLogMoneyness(_TestFeature):
    """
    pfhedge.features.LogMoneyness
    """

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test(self, strike):
        derivative = EuropeanOption(BrownianStock(), strike=strike)
        derivative.underlier.spot = torch.arange(1.0, 7.0).reshape(2, 3)
        # tensor([[1., 2., 3.],
        #         [4., 5., 6.]])
        f = LogMoneyness().of(derivative)

        result = f[0]
        expect = torch.tensor([[1.0], [4.0]]) / strike
        expect = torch.log(expect)
        assert_close(result, expect)
        result = f[1]
        expect = torch.tensor([[2.0], [5.0]]) / strike
        expect = torch.log(expect)
        assert_close(result, expect)
        result = f[2]
        expect = torch.tensor([[3.0], [6.0]]) / strike
        expect = torch.log(expect)
        assert_close(result, expect)

    def test_str(self):
        assert str(LogMoneyness()) == "log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock())
        self.assert_same_dtype(LogMoneyness(), derivative, dtype)


class TestExpiryTime(_TestFeature):
    """
    pfhedge.features.ExpiryTime
    """

    def test(self):
        derivative = EuropeanOption(BrownianStock(dt=0.1), maturity=0.2)
        derivative.underlier.spot = torch.empty(2, 3)
        f = ExpiryTime().of(derivative)

        result = f[0]
        expect = torch.full((2, 1), 0.2)
        assert_close(result, expect, check_stride=False)

        result = f[1]
        expect = torch.full((2, 1), 0.1)
        assert_close(result, expect, check_stride=False)

        result = f[2]
        expect = torch.full((2, 1), 0.0)
        assert_close(result, expect, check_stride=False)

    def test_2(self):
        derivative = EuropeanOption(BrownianStock(dt=0.1), maturity=0.15)
        derivative.underlier.spot = torch.empty(2, 3)
        f = ExpiryTime().of(derivative)

        result = f[0]
        expect = torch.full((2, 1), 0.2)
        assert_close(result, expect, check_stride=False)

        result = f[1]
        expect = torch.full((2, 1), 0.1)
        assert_close(result, expect, check_stride=False)

        result = f[2]
        expect = torch.full((2, 1), 0.0)
        assert_close(result, expect, check_stride=False)

    def test_str(self):
        assert str(ExpiryTime()) == "expiry_time"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock())
        self.assert_same_dtype(ExpiryTime(), derivative, dtype)


class TestVolatility(_TestFeature):
    """
    pfhedge.features.Volatility
    """

    @pytest.mark.parametrize("volatility", [0.2, 0.1])
    def test(self, volatility):
        derivative = EuropeanOption(BrownianStock(volatility=volatility))
        derivative.underlier.spot = torch.empty(2, 3)

        f = Volatility().of(derivative)

        result = f[0]
        expect = torch.full((2, 1), volatility)
        assert_close(result, expect)
        result = f[1]
        expect = torch.full((2, 1), volatility)
        assert_close(result, expect)
        result = f[2]
        expect = torch.full((2, 1), volatility)
        assert_close(result, expect)

    def test_stochastic_volatility(self):
        derivative = EuropeanOption(HestonStock())
        variance = torch.arange(6).resize(2, 3)
        derivative.ul().register_buffer("spot", torch.empty_like(variance))
        derivative.ul().register_buffer("variance", variance)

        f = Volatility().of(derivative)

        result = f[0]
        expect = variance[:, [0]].sqrt()
        assert_close(result, expect, check_stride=False)
        result = f[1]
        expect = variance[:, [1]].sqrt()
        assert_close(result, expect, check_stride=False)

    def test_str(self):
        assert str(Volatility()) == "volatility"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock())
        self.assert_same_dtype(Volatility(), derivative, dtype)

    def test_spot_and_volatility_are_not_the_same_size(self):
        derivative = EuropeanOption(HestonStock())
        derivative.simulate()
        derivative.ul().register_buffer("variance", torch.empty(1, 1))
        f = Volatility().of(derivative)
        with pytest.raises(ValueError):
            _ = f[0]


class TestPrevHedge(_TestFeature):
    """
    pfhedge.features.PrevHedge
    """

    @pytest.mark.parametrize("volatility", [0.2, 0.1])
    def test(self, volatility):
        torch.manual_seed(42)
        derivative = EuropeanOption(BrownianStock(volatility))
        derivative.underlier.spot = torch.randn(2, 3)
        hedger = Hedger(Linear(2, 1), ["moneyness", "expiry_time"])
        hedger.inputs = [feature.of(derivative) for feature in hedger.inputs]

        f = PrevHedge().of(derivative, hedger)

        result = f[0]
        expect = torch.zeros((2, 1))
        assert_close(result, expect)
        h = hedger(torch.cat([feature[0] for feature in hedger.inputs], 1))
        result = f[1]
        expect = h.reshape(-1, 1)
        assert_close(result, expect)
        h = hedger(torch.cat([feature[1] for feature in hedger.inputs], 1))
        result = f[2]
        expect = h.reshape(-1, 1)
        assert_close(result, expect)

    def test_str(self):
        assert str(PrevHedge()) == "prev_hedge"


class TestBarrier(_TestFeature):
    """
    pfhedge.inputs.Barrier
    """

    def test(self):
        derivative = EuropeanOption(BrownianStock())
        derivative.underlier.spot = torch.tensor(
            [
                [1.0, 1.5, 2.0, 3.0],
                [2.0, 1.0, 1.0, 1.0],
                [3.0, 4.0, 5.0, 6.0],
                [1.0, 1.1, 1.2, 1.3],
            ]
        )
        f = Barrier(2.0, up=True).of(derivative)

        result = f[0]
        expect = torch.tensor([0.0, 1.0, 1.0, 0.0]).reshape(-1, 1)
        assert_close(result, expect)
        result = f[1]
        expect = torch.tensor([0.0, 1.0, 1.0, 0.0]).reshape(-1, 1)
        assert_close(result, expect)
        result = f[2]
        expect = torch.tensor([1.0, 1.0, 1.0, 0.0]).reshape(-1, 1)
        assert_close(result, expect)
        result = f[3]
        expect = torch.tensor([1.0, 1.0, 1.0, 0.0]).reshape(-1, 1)
        assert_close(result, expect)

        derivative = EuropeanOption(BrownianStock())
        derivative.underlier.spot = torch.tensor(
            [
                [3.0, 2.0, 1.5, 1.0],
                [1.0, 1.0, 1.0, 2.0],
                [6.0, 5.0, 4.0, 3.0],
                [1.3, 1.2, 1.1, 1.0],
            ]
        )
        f = Barrier(2.0, up=False).of(derivative)

        result = f[0]
        expect = torch.tensor([0.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
        assert_close(result, expect)
        result = f[1]
        expect = torch.tensor([1.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
        assert_close(result, expect)
        result = f[2]
        expect = torch.tensor([1.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
        assert_close(result, expect)
        result = f[3]
        expect = torch.tensor([1.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
        assert_close(result, expect)

    def test_repr(self):
        assert repr(Barrier(1.0, up=True)) == "Barrier(1.0, up=True)"
        assert repr(Barrier(2.0, up=True)) == "Barrier(2.0, up=True)"
        assert repr(Barrier(1.0, up=False)) == "Barrier(1.0, up=False)"
        assert repr(Barrier(2.0, up=False)) == "Barrier(2.0, up=False)"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock())
        self.assert_same_dtype(Barrier(1.0), derivative, dtype)


class TestZeros(_TestFeature):
    """
    pfhedge.features.Zeros
    """

    def test(self):
        torch.manual_seed(42)
        derivative = EuropeanOption(BrownianStock())
        derivative.underlier.spot = torch.empty(2, 3)

        f = Zeros().of(derivative)

        result = f[0]
        expect = torch.zeros((2, 1))
        assert_close(result, expect)
        result = f[1]
        expect = torch.zeros((2, 1))
        assert_close(result, expect)
        result = f[2]
        expect = torch.zeros((2, 1))
        assert_close(result, expect)

    def test_str(self):
        assert str(Zeros()) == "zeros"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock())
        self.assert_same_dtype(Zeros(), derivative, dtype)


class TestEmpty(_TestFeature):
    """
    pfhedge.features.Zeros
    """

    def test(self):
        torch.manual_seed(42)
        derivative = EuropeanOption(BrownianStock())
        derivative.underlier.spot = torch.empty(2, 3)

        f = Empty().of(derivative)

        result = f[0]
        assert result.size() == torch.Size((2, 1))
        result = f[1]
        assert result.size() == torch.Size((2, 1))
        result = f[2]
        assert result.size() == torch.Size((2, 1))

    def test_str(self):
        assert str(Empty()) == "empty"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock())
        self.assert_same_dtype(Empty(), derivative, dtype)


class TestMaxMoneyness(_TestFeature):
    """
    pfhedge.features.MaxMoneyness
    """

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test(self, strike, log):
        derivative = EuropeanOption(BrownianStock(), strike=strike)
        derivative.underlier.spot = torch.tensor(
            [[1.0, 2.0, 1.5], [2.0, 3.0, 4.0], [3.0, 2.0, 1.0]]
        )

        f = MaxMoneyness(log=log).of(derivative)

        result = f[0]
        expect = torch.tensor([[1.0], [2.0], [3.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert_close(result, expect)
        result = f[1]
        expect = torch.tensor([[2.0], [3.0], [3.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert_close(result, expect)
        result = f[2]
        expect = torch.tensor([[2.0], [4.0], [3.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert_close(result, expect)

    def test_str(self):
        assert str(MaxMoneyness()) == "max_moneyness"
        assert str(MaxMoneyness(log=True)) == "max_log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock())
        self.assert_same_dtype(MaxMoneyness(), derivative, dtype)


class TestMaxLogMoneyness(_TestFeature):
    """
    pfhedge.features.MaxLogMoneyness
    """

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test(self, strike):
        derivative = EuropeanOption(BrownianStock(), strike=strike)
        derivative.underlier.spot = torch.tensor(
            [[1.0, 2.0, 1.5], [2.0, 3.0, 4.0], [3.0, 2.0, 1.0]]
        )

        f = MaxLogMoneyness().of(derivative)

        result = f[0]
        expect = torch.tensor([[1.0], [2.0], [3.0]]) / strike
        expect = torch.log(expect)
        assert_close(result, expect)
        result = f[1]
        expect = torch.tensor([[2.0], [3.0], [3.0]]) / strike
        expect = torch.log(expect)
        assert_close(result, expect)
        result = f[2]
        expect = torch.tensor([[2.0], [4.0], [3.0]]) / strike
        expect = torch.log(expect)
        assert_close(result, expect)

    def test_str(self):
        assert str(MaxLogMoneyness()) == "max_log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock())
        self.assert_same_dtype(MaxLogMoneyness(), derivative, dtype)


class TestModelOutput(_TestFeature):
    def test(self):
        derivative = EuropeanOption(BrownianStock())
        derivative.simulate()

        module = torch.nn.Linear(2, 1)
        x1, x2 = Moneyness(), ExpiryTime()
        f = ModuleOutput(module, [x1, x2]).of(derivative)

        result = f[0]
        expect = module(torch.cat([x1[0], x2[0]], 1))
        assert_close(result, expect)
        result = f[1]
        expect = module(torch.cat([x1[1], x2[1]], 1))
        assert_close(result, expect)
        result = f[2]
        expect = module(torch.cat([x1[2], x2[2]], 1))
        assert_close(result, expect)

    def test_repr(self):
        module = torch.nn.Linear(2, 1)
        x1, x2 = Moneyness(), ExpiryTime()
        f = ModuleOutput(module, [x1, x2])
        expect = (
            "ModuleOutput(\n"
            "  inputs=['moneyness', 'expiry_time']\n"
            "  (module): Linear(in_features=2, out_features=1, bias=True)\n"
            ")"
        )
        assert repr(f) == expect

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = EuropeanOption(BrownianStock()).to(dtype)
        m = Linear(2, 1).to(derivative.dtype)
        f = ModuleOutput(m, [Moneyness(), ExpiryTime()])
        self.assert_same_dtype(f, derivative, dtype)
