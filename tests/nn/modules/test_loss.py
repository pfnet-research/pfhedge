import pytest
import torch
from torch.testing import assert_close

from pfhedge._utils.testing import assert_cash_equivalent
from pfhedge._utils.testing import assert_cash_invariant
from pfhedge._utils.testing import assert_convex
from pfhedge._utils.testing import assert_monotone
from pfhedge.nn import EntropicLoss
from pfhedge.nn import EntropicRiskMeasure
from pfhedge.nn import ExpectedShortfall
from pfhedge.nn import IsoelasticLoss
from pfhedge.nn.modules.loss import OCE
from pfhedge.nn.modules.loss import QuadraticCVaR


def assert_loss_shape(loss):
    torch.manual_seed(42)

    N = 20
    M_1 = 10
    M_2 = 11

    input = torch.randn((N,)).exp()  # impose > 0
    output = loss(input)
    assert output.size() == torch.Size([])
    output = loss.cash(input)
    assert output.size() == torch.Size([])

    input = torch.randn((N, M_1)).exp()
    output = loss(input)
    assert output.size() == torch.Size((M_1,))
    output = loss.cash(input)
    assert output.size() == torch.Size((M_1,))

    input = torch.randn((N, M_1, M_2)).exp()
    output = loss(input)
    assert output.size() == torch.Size((M_1, M_2))
    output = loss.cash(input)
    assert output.size() == torch.Size((M_1, M_2))


class TestEntropicRiskMeasure:
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(self, n_paths, risk, a):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(risk)
        x1 = torch.randn(n_paths)
        x2 = x1 - a
        assert_monotone(loss, x1, x2, increasing=False)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(self, n_paths, risk, a):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(risk)
        x1 = torch.randn(n_paths)
        x2 = torch.randn(n_paths)
        assert_convex(loss, x1, x2, a)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_cash(self, n_paths, a):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(a)
        x = torch.randn(n_paths)
        assert_cash_equivalent(loss, x, float(loss.cash(x).item()))

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("c", [0.001, 1, 2])
    def test_cash_equivalent(self, n_paths, risk, c):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(risk)
        assert_cash_invariant(loss, torch.randn(n_paths), c)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_value(self, n_paths, a):
        torch.manual_seed(42)

        value = 1.0
        loss = EntropicRiskMeasure(a)
        result = loss(torch.full((n_paths,), value))
        expect = torch.log(torch.exp(-a * torch.tensor(value))) / a
        assert_close(result, expect)

    def test_error_a(self):
        with pytest.raises(ValueError):
            EntropicRiskMeasure(0)
        with pytest.raises(ValueError):
            EntropicRiskMeasure(-1)

    def test_repr(self):
        loss = EntropicRiskMeasure()
        assert repr(loss) == "EntropicRiskMeasure()"
        loss = EntropicRiskMeasure(a=10.0)
        assert repr(loss) == "EntropicRiskMeasure(a=10.)"

    def test_shape(self):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure()
        assert_loss_shape(loss)


class TestEntropicLoss:
    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(self, n_paths, risk, a):
        torch.manual_seed(42)

        loss = EntropicLoss(risk)
        x1 = torch.randn(n_paths)
        x2 = x1 - a
        assert_monotone(loss, x1, x2, increasing=False)

    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(self, n_paths, risk, a):
        torch.manual_seed(42)

        loss = EntropicLoss(risk)
        input1 = torch.randn(n_paths)
        input2 = torch.randn(n_paths)
        assert_convex(loss, input1, input2, a)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_cash(self, n_paths, a):
        torch.manual_seed(42)

        loss = EntropicLoss(a)
        x = torch.randn(n_paths)
        assert_cash_equivalent(loss, x, float(loss.cash(x).item()))

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_value(self, n_paths, a):
        value = 1.0
        loss = EntropicLoss(a)
        result = loss(torch.full((n_paths,), value))
        expect = torch.exp(-a * torch.tensor(value))
        assert_close(result, expect)

    def test_error_a(self):
        with pytest.raises(ValueError):
            EntropicLoss(0)
        with pytest.raises(ValueError):
            EntropicLoss(-1)

    def test_repr(self):
        loss = EntropicLoss()
        assert repr(loss) == "EntropicLoss()"
        loss = EntropicLoss(a=10.0)
        assert repr(loss) == "EntropicLoss(a=10.)"

    def test_shape(self):
        torch.manual_seed(42)

        loss = EntropicLoss()
        assert_loss_shape(loss)


class TestIsoelasticLoss:
    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(self, n_paths, risk, a):
        torch.manual_seed(42)

        loss = IsoelasticLoss(risk)
        x2 = torch.randn(n_paths).exp()  # force positive
        x1 = x2 + a

        assert_monotone(loss, x1, x2, increasing=False)

    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(self, n_paths, risk, a):
        torch.manual_seed(42)

        loss = IsoelasticLoss(risk)
        x1 = torch.randn(n_paths).exp()
        x2 = torch.randn(n_paths).exp()
        assert_convex(loss, x1, x2, a)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    def test_cash(self, n_paths, risk):
        torch.manual_seed(42)

        loss = IsoelasticLoss(risk)
        x = torch.randn(n_paths).exp()  # force positive
        assert_cash_equivalent(loss, x, float(loss.cash(x).item()))

    def test_error_a(self):
        with pytest.raises(ValueError):
            IsoelasticLoss(0.0)
        with pytest.raises(ValueError):
            IsoelasticLoss(-1.0)
        with pytest.raises(ValueError):
            IsoelasticLoss(2.0)

    def test_repr(self):
        loss = IsoelasticLoss(0.5)
        assert repr(loss) == "IsoelasticLoss(a=0.5000)"

    def test_shape(self):
        torch.manual_seed(42)

        loss = IsoelasticLoss(0.5)
        assert_loss_shape(loss)

        loss = IsoelasticLoss(1.0)
        assert_loss_shape(loss)


class TestExpectedShortFall:
    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.5])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(self, n_paths, p, a):
        torch.manual_seed(42)

        loss = ExpectedShortfall(p)
        x1 = torch.randn(n_paths)
        x2 = x1 - 1
        assert_monotone(loss, x1, x2, increasing=False)

    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(self, n_paths, p, a):
        torch.manual_seed(42)

        loss = ExpectedShortfall(p)
        x1 = torch.randn(n_paths)
        x2 = torch.randn(n_paths)
        assert_convex(loss, x1, x2, a)

    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_cash(self, n_paths, p):
        torch.manual_seed(42)

        loss = ExpectedShortfall(p)
        x = torch.randn(n_paths)
        assert_cash_equivalent(loss, x, float(loss.cash(x).item()))

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("eta", [0.001, 1, 2])
    def test_cash_equivalent(self, n_paths, p, eta):
        loss = ExpectedShortfall(p)
        assert_cash_invariant(loss, torch.randn(n_paths), eta)

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
        torch.manual_seed(42)

        n_paths = 100
        k = int(n_paths * percentile)
        loss = ExpectedShortfall(percentile)

        input = torch.randn(n_paths)

        result = loss(input)
        expect = -torch.mean(torch.tensor(sorted(input)[:k]))

        assert_close(result, expect)

    def test_repr(self):
        loss = ExpectedShortfall(0.1)
        assert repr(loss) == "ExpectedShortfall(0.1)"
        loss = ExpectedShortfall(0.5)
        assert repr(loss) == "ExpectedShortfall(0.5)"

    def test_shape(self):
        torch.manual_seed(42)

        loss = ExpectedShortfall()
        assert_loss_shape(loss)


class TestQuadraticCVaR:
    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("lam", [1.0, 2.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(self, n_paths, lam, a):
        torch.manual_seed(42)

        loss = QuadraticCVaR(lam)
        x1 = torch.randn(n_paths)
        x2 = x1 - 1
        assert_monotone(loss, x1, x2, increasing=False)

    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("lam", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(self, n_paths, lam, a):
        torch.manual_seed(42)

        loss = QuadraticCVaR(lam)
        x1 = torch.randn(n_paths)
        x2 = torch.randn(n_paths)
        assert_convex(loss, x1, x2, a)

    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("lam", [1.0, 2.0, 10.0])
    def test_cash(self, n_paths, lam):
        torch.manual_seed(42)

        loss = QuadraticCVaR(lam)
        x = torch.randn(n_paths)
        assert_cash_equivalent(loss, x, float(loss.cash(x).item()))

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("lam", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("eta", [0.001, 1, 2])
    def test_cash_equivalent(self, n_paths, lam, eta):
        loss = QuadraticCVaR(lam)
        assert_cash_invariant(loss, torch.randn(n_paths), eta)

    def test_error_percentile(self):
        # 1 is allowed
        _ = QuadraticCVaR(1.0)
        with pytest.raises(ValueError):
            _ = QuadraticCVaR(0)
        with pytest.raises(ValueError):
            _ = QuadraticCVaR(-1)
        with pytest.raises(ValueError):
            _ = QuadraticCVaR(0.9)

    @pytest.mark.parametrize("percentile", [0.1, 0.5, 0.9])
    def test_value(self, percentile):
        torch.manual_seed(42)

        n_paths = 100
        k = int(n_paths * percentile)
        loss = ExpectedShortfall(percentile)

        input = torch.randn(n_paths)

        result = loss(input)
        expect = -torch.mean(torch.tensor(sorted(input)[:k]))

        assert_close(result, expect)

    def test_repr(self):
        loss = QuadraticCVaR(1.0)
        assert repr(loss) == "QuadraticCVaR(1.0)"
        loss = QuadraticCVaR(2.0)
        assert repr(loss) == "QuadraticCVaR(2.0)"

    def test_shape(self):
        torch.manual_seed(42)

        loss = QuadraticCVaR()
        assert_loss_shape(loss)


class TestOCE:
    def train_oce(self, m):
        torch.manual_seed(42)

        optim = torch.optim.Adam(m.parameters())

        for _ in range(1000):
            optim.zero_grad()
            m(torch.randn(10000)).backward()
            optim.step()

    def test_fit(self):
        torch.manual_seed(42)

        m = OCE(lambda input: 1 - torch.exp(-input))

        self.train_oce(m)

        input = torch.randn(10000)

        result = m(input)
        expect = torch.log(EntropicLoss()(input))

        assert_close(result, expect, rtol=1e-02, atol=1e-5)

    def test_repr(self):
        def exp_utility(input):
            return 1 - torch.exp(-input)

        loss = OCE(exp_utility)
        assert repr(loss) == "OCE(exp_utility, w=0.)"

    def test_shape(self):
        torch.manual_seed(42)

        loss = OCE(lambda input: 1 - torch.exp(-input))
        assert_loss_shape(loss)
