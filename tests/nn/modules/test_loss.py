import pytest
import torch

from pfhedge.nn import EntropicLoss
from pfhedge.nn import EntropicRiskMeasure
from pfhedge.nn import ExpectedShortfall
from pfhedge.nn import IsoelasticLoss
from pfhedge.nn.modules.loss import OCE


class _TestHedgeLoss:
    def assert_nonincreasing(self, loss, x, a):
        assert a > 0
        ll = loss(x).item()
        lu = loss(x + a).item()
        assert ll > lu

    def assert_convex(self, loss, x1, x2, a):
        ll = loss(a * x1 + (1 - a) * x2).item()
        lu = (a * loss(x1) + (1 - a) * loss(x2)).item()
        assert ll < lu

    def assert_cash_equivalent(self, loss, x, eta):
        result = loss(x + eta)
        expect = loss(x) - eta
        assert torch.isclose(result, expect)

    def assert_cash(self, loss, x):
        result = loss(x)
        expect = loss(torch.full_like(x, loss.cash(x)))
        assert torch.isclose(result, expect)

    def assert_shape(self, loss):
        torch.manual_seed(42)

        N = 20
        M_1 = 10
        M_2 = 11

        x = torch.randn((N,)).exp()  # impose > 0
        out = loss(x)
        assert out.size() == torch.Size([])
        out = loss.cash(x)
        assert out.size() == torch.Size([])

        x = torch.randn((N, M_1)).exp()
        out = loss(x)
        assert out.size() == torch.Size((M_1,))
        out = loss.cash(x)

        x = torch.randn((N, M_1, M_2)).exp()
        out = loss(x)
        assert out.size() == torch.Size((M_1, M_2))
        out = loss.cash(x)
        assert out.size() == torch.Size((M_1, M_2))


class TestEntropicRiskMeasure(_TestHedgeLoss):
    """
    pfhedge.nn.EntropicRiskMeasure
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(self, n_paths, risk, a):
        loss = EntropicRiskMeasure(risk)
        self.assert_nonincreasing(loss, torch.randn(n_paths), a)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(self, n_paths, risk, a):
        loss = EntropicRiskMeasure(risk)
        x1 = torch.randn(n_paths)
        x2 = torch.randn(n_paths)
        self.assert_convex(loss, x1, x2, a)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_cash(self, n_paths, a):
        loss = EntropicRiskMeasure(a)
        self.assert_cash(loss, torch.randn(n_paths))

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("eta", [0.001, 1, 2])
    def test_cash_equivalent(self, n_paths, risk, eta):
        loss = EntropicRiskMeasure(risk)
        self.assert_cash_equivalent(loss, torch.randn(n_paths), eta)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_value(self, n_paths, a):
        value = 1.0
        loss = EntropicRiskMeasure(a)
        result = loss(torch.full((n_paths,), value))
        expect = torch.log(torch.exp(-a * torch.tensor(value))) / a
        assert torch.isclose(result, expect)

    def test_error_a(self):
        with pytest.raises(ValueError):
            EntropicRiskMeasure(0)
        with pytest.raises(ValueError):
            EntropicRiskMeasure(-1)

    def test_repr(self):
        loss = EntropicRiskMeasure()
        assert repr(loss) == "EntropicRiskMeasure()"
        loss = EntropicRiskMeasure(a=10.0)
        assert repr(loss) == "EntropicRiskMeasure(a=10.0)"

    def test_shape(self):
        loss = EntropicRiskMeasure()
        self.assert_shape(loss)


class TestEntropicLoss(_TestHedgeLoss):
    """
    pfhedge.nn.EntropicLoss
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(self, n_paths, risk, a):
        loss = EntropicLoss(risk)
        self.assert_nonincreasing(loss, torch.randn(n_paths), a)

    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(self, n_paths, risk, a):
        loss = EntropicLoss(risk)
        x1 = torch.randn(n_paths)
        x2 = torch.randn(n_paths)
        self.assert_convex(loss, x1, x2, a)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_cash(self, n_paths, a):
        loss = EntropicLoss(a)
        self.assert_cash(loss, torch.randn(n_paths))

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_value(self, n_paths, a):
        value = 1.0
        loss = EntropicLoss(a)
        result = loss(torch.full((n_paths,), value))
        expect = torch.exp(-a * torch.tensor(value))
        assert torch.isclose(result, expect)

    def test_error_a(self):
        with pytest.raises(ValueError):
            EntropicLoss(0)
        with pytest.raises(ValueError):
            EntropicLoss(-1)

    def test_repr(self):
        loss = EntropicLoss()
        assert repr(loss) == "EntropicLoss()"
        loss = EntropicLoss(a=10.0)
        assert repr(loss) == "EntropicLoss(a=10.0)"

    def test_shape(self):
        loss = EntropicLoss()
        self.assert_shape(loss)


class TestIsoelasticLoss(_TestHedgeLoss):
    """
    pfhedge.nn.IsoelasticLoss
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(self, n_paths, risk, a):
        loss = IsoelasticLoss(risk)
        x = torch.exp(torch.randn(n_paths))  # force positive
        self.assert_nonincreasing(loss, x, a)

    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(self, n_paths, risk, a):
        loss = IsoelasticLoss(risk)
        x1 = torch.exp(torch.randn(n_paths))
        x2 = torch.exp(torch.randn(n_paths))
        self.assert_convex(loss, x1, x2, a)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    def test_cash(self, n_paths, risk):
        loss = IsoelasticLoss(risk)
        x = torch.exp(torch.randn(n_paths))  # force positive
        self.assert_cash(loss, x)

    def test_error_a(self):
        with pytest.raises(ValueError):
            IsoelasticLoss(0.0)
        with pytest.raises(ValueError):
            IsoelasticLoss(-1.0)
        with pytest.raises(ValueError):
            IsoelasticLoss(2.0)

    def test_repr(self):
        loss = IsoelasticLoss(0.5)
        assert repr(loss) == "IsoelasticLoss(a=0.5)"

    def test_shape(self):
        loss = IsoelasticLoss(0.5)
        self.assert_shape(loss)
        loss = IsoelasticLoss(1.0)
        self.assert_shape(loss)


class TestExpectedShortFall(_TestHedgeLoss):
    """
    pfhedge.nn.ExpectedShortFall
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.5])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(self, n_paths, p, a):
        loss = ExpectedShortfall(p)
        x = torch.randn(n_paths)
        self.assert_nonincreasing(loss, x, a)

    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(self, n_paths, p, a):
        loss = ExpectedShortfall(p)
        x1 = torch.randn(n_paths)
        x2 = torch.randn(n_paths)
        self.assert_convex(loss, x1, x2, a)

    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_cash(self, n_paths, p):
        loss = ExpectedShortfall(p)
        x = torch.randn(n_paths)
        self.assert_cash(loss, x)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("eta", [0.001, 1, 2])
    def test_cash_equivalent(self, n_paths, p, eta):
        loss = ExpectedShortfall(p)
        self.assert_cash_equivalent(loss, torch.randn(n_paths), eta)

    def test_error_percentile(self):
        # 1 is allowed
        _ = ExpectedShortfall(1)
        with pytest.raises(ValueError):
            _ = ExpectedShortfall(0)
        with pytest.raises(ValueError):
            _ = ExpectedShortfall(-1)
        with pytest.raises(ValueError):
            _ = ExpectedShortfall(1.1)

    @pytest.mark.parametrize("percentile", [0.1, 0.5, 0.9])
    def test_value(self, percentile):
        n_paths = 100
        k = int(n_paths * percentile)
        loss = ExpectedShortfall(percentile)

        x = torch.randn(n_paths)

        result = loss(x)
        expect = -torch.mean(torch.tensor(sorted(x)[:k]))

        assert torch.isclose(result, expect)

    def test_repr(self):
        loss = ExpectedShortfall(0.1)
        assert repr(loss) == "ExpectedShortfall(0.1)"
        loss = ExpectedShortfall(0.5)
        assert repr(loss) == "ExpectedShortfall(0.5)"

    def test_shape(self):
        loss = ExpectedShortfall()
        self.assert_shape(loss)


class TestOCE(_TestHedgeLoss):
    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def train_oce(self, m):
        optim = torch.optim.Adam(m.parameters())

        for _ in range(1000):
            optim.zero_grad()
            m(torch.randn(10000)).backward()
            optim.step()

    def test_fit(self):
        torch.manual_seed(42)
        m = OCE(lambda x: 1 - torch.exp(-x))

        self.train_oce(m)

        x = torch.randn(10000)

        result = m(x)
        expect = torch.log(EntropicLoss()(x))

        assert torch.isclose(result, expect, rtol=1e-02)

    def test_repr(self):
        def exp_utility(x):
            return 1 - torch.exp(-x)

        loss = OCE(exp_utility)
        assert repr(loss) == "OCE(exp_utility, w=0.0)"

    def test_shape(self):
        loss = OCE(lambda x: 1 - torch.exp(-x))
        self.assert_shape(loss)
