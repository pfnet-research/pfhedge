import pytest
import torch
from torch.nn import Linear

from pfhedge import Hedger
from pfhedge.features import Barrier
from pfhedge.features import ExpiryTime
from pfhedge.features import LogMoneyness
from pfhedge.features import MaxLogMoneyness
from pfhedge.features import MaxMoneyness
from pfhedge.features import ModuleOutput
from pfhedge.features import Moneyness
from pfhedge.features import PrevHedge
from pfhedge.features import Volatility
from pfhedge.features import Zero
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption


class _TestFeature:
    def assert_same_dtype(self, feature, liability, dtype):
        liability.to(dtype).simulate()
        assert feature.of(liability)[0].dtype == dtype


class TestMoneyness(_TestFeature):
    """
    pfhedge.features.Moneyness
    """

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test(self, strike, log):
        liability = EuropeanOption(BrownianStock(), strike=strike)
        liability.underlier.prices = torch.arange(1.0, 7.0).reshape(3, 2)
        # tensor([[1., 2.],
        #         [3., 4.],
        #         [5., 6.]])
        f = Moneyness(log=log).of(liability)

        result = f[0]
        expect = torch.tensor([[1.0], [2.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert torch.allclose(result, expect)
        result = f[1]
        expect = torch.tensor([[3.0], [4.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert torch.allclose(result, expect)
        result = f[2]
        expect = torch.tensor([[5.0], [6.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert torch.allclose(result, expect)

    def test_str(self):
        assert str(Moneyness()) == "moneyness"
        assert str(Moneyness(log=True)) == "log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock())
        self.assert_same_dtype(Moneyness(), liability, dtype)


class TestLogMoneyness(_TestFeature):
    """
    pfhedge.features.LogMoneyness
    """

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test(self, strike):
        liability = EuropeanOption(BrownianStock(), strike=strike)
        liability.underlier.prices = torch.arange(1.0, 7.0).reshape(3, 2)
        # tensor([[1., 2.],
        #         [3., 4.],
        #         [5., 6.]])
        f = LogMoneyness().of(liability)

        result = f[0]
        expect = torch.tensor([[1.0], [2.0]]) / strike
        expect = torch.log(expect)
        assert torch.allclose(result, expect)
        result = f[1]
        expect = torch.tensor([[3.0], [4.0]]) / strike
        expect = torch.log(expect)
        assert torch.allclose(result, expect)
        result = f[2]
        expect = torch.tensor([[5.0], [6.0]]) / strike
        expect = torch.log(expect)
        assert torch.allclose(result, expect)

    def test_str(self):
        assert str(LogMoneyness()) == "log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock())
        self.assert_same_dtype(LogMoneyness(), liability, dtype)


class TestExpiryTime(_TestFeature):
    """
    pfhedge.features.ExpiryTime
    """

    def test(self):
        maturity = 3 / 365
        dt = 1 / 365
        liability = EuropeanOption(BrownianStock(dt=dt), maturity=maturity)
        liability.underlier.prices = torch.arange(1.0, 7.0).reshape(3, 2)

        f = ExpiryTime().of(liability)

        result = f[0]
        expect = torch.full((2, 1), 3 / 365)
        assert torch.allclose(result, expect)
        result = f[1]
        expect = torch.full((2, 1), 2 / 365)
        assert torch.allclose(result, expect)
        result = f[2]
        expect = torch.full((2, 1), 1 / 365)
        assert torch.allclose(result, expect)

    def test_str(self):
        assert str(ExpiryTime()) == "expiry_time"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock())
        self.assert_same_dtype(ExpiryTime(), liability, dtype)


class TestVolatility(_TestFeature):
    """
    pfhedge.features.Volatility
    """

    @pytest.mark.parametrize("volatility", [0.2, 0.1])
    def test(self, volatility):
        liability = EuropeanOption(BrownianStock(volatility=volatility))
        liability.underlier.prices = torch.arange(1.0, 7.0).reshape(3, 2)

        f = Volatility().of(liability)

        result = f[0]
        expect = torch.full((2, 1), volatility)
        assert torch.allclose(result, expect)
        result = f[1]
        expect = torch.full((2, 1), volatility)
        assert torch.allclose(result, expect)
        result = f[2]
        expect = torch.full((2, 1), volatility)
        assert torch.allclose(result, expect)

    def test_str(self):
        assert str(Volatility()) == "volatility"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock())
        self.assert_same_dtype(Volatility(), liability, dtype)


class TestPrevHedge(_TestFeature):
    """
    pfhedge.features.PrevHedge
    """

    @pytest.mark.parametrize("volatility", [0.2, 0.1])
    def test(self, volatility):
        torch.manual_seed(42)
        liability = EuropeanOption(BrownianStock(volatility))
        liability.underlier.prices = torch.arange(1.0, 7.0).reshape(3, 2)
        hedger = Hedger(Linear(2, 1), ["moneyness", "expiry_time"])
        hedger.features = [feature.of(liability) for feature in hedger.features]

        f = PrevHedge().of(liability, hedger)

        result = f[0]
        expect = torch.zeros((2, 1))
        assert torch.allclose(result, expect)
        h = hedger(torch.cat([feature[0] for feature in hedger.features], 1))
        result = f[1]
        expect = h.reshape(-1, 1)
        assert torch.allclose(result, expect)
        h = hedger(torch.cat([feature[1] for feature in hedger.features], 1))
        result = f[2]
        expect = h.reshape(-1, 1)
        assert torch.allclose(result, expect)

    def test_str(self):
        assert str(PrevHedge()) == "prev_hedge"


class TestBarrier(_TestFeature):
    """
    pfhedge.features.Barrier
    """

    def test(self):
        liability = EuropeanOption(BrownianStock())
        liability.underlier.prices = torch.tensor(
            [
                [1.0, 2.0, 3.0, 1.0],
                [1.5, 1.0, 4.0, 1.1],
                [2.0, 1.0, 5.0, 1.2],
                [3.0, 1.0, 6.0, 1.3],
            ]
        )
        f = Barrier(2.0, up=True).of(liability)

        result = f[0]
        expect = torch.tensor([0.0, 1.0, 1.0, 0.0]).reshape(-1, 1)
        assert torch.allclose(result, expect)
        result = f[1]
        expect = torch.tensor([0.0, 1.0, 1.0, 0.0]).reshape(-1, 1)
        assert torch.allclose(result, expect)
        result = f[2]
        expect = torch.tensor([1.0, 1.0, 1.0, 0.0]).reshape(-1, 1)
        assert torch.allclose(result, expect)
        result = f[3]
        expect = torch.tensor([1.0, 1.0, 1.0, 0.0]).reshape(-1, 1)
        assert torch.allclose(result, expect)

        liability = EuropeanOption(BrownianStock())
        liability.underlier.prices = torch.tensor(
            [
                [3.0, 1.0, 6.0, 1.3],
                [2.0, 1.0, 5.0, 1.2],
                [1.5, 1.0, 4.0, 1.1],
                [1.0, 2.0, 3.0, 1.0],
            ]
        )
        f = Barrier(2.0, up=False).of(liability)

        result = f[0]
        expect = torch.tensor([0.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
        assert torch.allclose(result, expect)
        result = f[1]
        expect = torch.tensor([1.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
        assert torch.allclose(result, expect)
        result = f[2]
        expect = torch.tensor([1.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
        assert torch.allclose(result, expect)
        result = f[3]
        expect = torch.tensor([1.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
        assert torch.allclose(result, expect)

    def test_repr(self):
        assert repr(Barrier(1.0, up=True)) == "Barrier(1.0, up=True)"
        assert repr(Barrier(2.0, up=True)) == "Barrier(2.0, up=True)"
        assert repr(Barrier(1.0, up=False)) == "Barrier(1.0, up=False)"
        assert repr(Barrier(2.0, up=False)) == "Barrier(2.0, up=False)"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock())
        self.assert_same_dtype(Barrier(1.0), liability, dtype)


class TestZero(_TestFeature):
    """
    pfhedge.features.Zero
    """

    def test(self):
        torch.manual_seed(42)
        liability = EuropeanOption(BrownianStock())
        liability.underlier.prices = torch.arange(1.0, 7.0).reshape(3, 2)

        f = Zero().of(liability)

        result = f[0]
        expect = torch.zeros((2, 1))
        assert torch.allclose(result, expect)
        result = f[1]
        expect = torch.zeros((2, 1))
        assert torch.allclose(result, expect)
        result = f[2]
        expect = torch.zeros((2, 1))
        assert torch.allclose(result, expect)

    def test_str(self):
        assert str(Zero()) == "zero"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock())
        self.assert_same_dtype(Zero(), liability, dtype)


class TestMaxMoneyness(_TestFeature):
    """
    pfhedge.features.MaxMoneyness
    """

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    @pytest.mark.parametrize("log", [True, False])
    def test(self, strike, log):
        liability = EuropeanOption(BrownianStock(), strike=strike)
        liability.underlier.prices = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 2.0],
                [1.5, 4.0, 1.0],
            ]
        )

        f = MaxMoneyness(log=log).of(liability)

        result = f[0]
        expect = torch.tensor([[1.0], [2.0], [3.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert torch.allclose(result, expect)
        result = f[1]
        expect = torch.tensor([[2.0], [3.0], [3.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert torch.allclose(result, expect)
        result = f[2]
        expect = torch.tensor([[2.0], [4.0], [3.0]]) / strike
        expect = torch.log(expect) if log else expect
        assert torch.allclose(result, expect)

    def test_str(self):
        assert str(MaxMoneyness()) == "max_moneyness"
        assert str(MaxMoneyness(log=True)) == "max_log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock())
        self.assert_same_dtype(MaxMoneyness(), liability, dtype)


class TestMaxLogMoneyness(_TestFeature):
    """
    pfhedge.features.MaxLogMoneyness
    """

    @pytest.mark.parametrize("strike", [1.0, 2.0])
    def test(self, strike):
        liability = EuropeanOption(BrownianStock(), strike=strike)
        liability.underlier.prices = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 2.0],
                [1.5, 4.0, 1.0],
            ]
        )

        f = MaxLogMoneyness().of(liability)

        result = f[0]
        expect = torch.tensor([[1.0], [2.0], [3.0]]) / strike
        expect = torch.log(expect)
        assert torch.allclose(result, expect)
        result = f[1]
        expect = torch.tensor([[2.0], [3.0], [3.0]]) / strike
        expect = torch.log(expect)
        assert torch.allclose(result, expect)
        result = f[2]
        expect = torch.tensor([[2.0], [4.0], [3.0]]) / strike
        expect = torch.log(expect)
        assert torch.allclose(result, expect)

    def test_str(self):
        assert str(MaxLogMoneyness()) == "max_log_moneyness"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock())
        self.assert_same_dtype(MaxLogMoneyness(), liability, dtype)


class TestModelOutput(_TestFeature):
    def test(self):
        liability = EuropeanOption(BrownianStock())
        liability.simulate()

        module = torch.nn.Linear(2, 1)
        x1, x2 = Moneyness(), ExpiryTime()
        f = ModuleOutput(module, [x1, x2]).of(liability)

        result = f[0]
        expect = module(torch.cat([x1[0], x2[0]], 1))
        assert torch.allclose(result, expect)
        result = f[1]
        expect = module(torch.cat([x1[1], x2[1]], 1))
        assert torch.allclose(result, expect)
        result = f[2]
        expect = module(torch.cat([x1[2], x2[2]], 1))
        assert torch.allclose(result, expect)

    def test_repr(self):
        module = torch.nn.Linear(2, 1)
        x1, x2 = Moneyness(), ExpiryTime()
        f = ModuleOutput(module, [x1, x2])
        expect = (
            "ModuleOutput(\n"
            "  features=['moneyness', 'expiry_time'],\n"
            "  (module): Linear(in_features=2, out_features=1, bias=True)\n"
            ")"
        )
        assert repr(f) == expect

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        liability = EuropeanOption(BrownianStock()).to(dtype)
        m = Linear(2, 1).to(liability.dtype)
        f = ModuleOutput(m, [Moneyness(), ExpiryTime()])
        self.assert_same_dtype(f, liability, dtype)
