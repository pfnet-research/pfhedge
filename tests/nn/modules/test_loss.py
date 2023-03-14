from typing import Optional
from typing import Union

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


def assert_loss_shape(loss, device: Optional[Union[str, torch.device]] = "cpu"):
    torch.manual_seed(42)

    N = 20
    M_1 = 10
    M_2 = 11

    input = torch.randn((N,)).to(device).exp()  # impose > 0
    output = loss(input)
    assert output.size() == torch.Size([])
    output = loss.cash(input)
    assert output.size() == torch.Size([])

    input = torch.randn((N, M_1)).to(device).exp()
    output = loss(input)
    assert output.size() == torch.Size((M_1,))
    output = loss.cash(input)
    assert output.size() == torch.Size((M_1,))

    input = torch.randn((N, M_1, M_2)).to(device).exp()
    output = loss(input)
    assert output.size() == torch.Size((M_1, M_2))
    output = loss.cash(input)
    assert output.size() == torch.Size((M_1, M_2))


class TestEntropicRiskMeasure:
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(
        self, n_paths, risk, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(risk).to(device)
        x1 = torch.randn(n_paths).to(device)
        x2 = x1 - a
        assert_monotone(loss, x1, x2, increasing=False)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing_gpu(self, n_paths, risk, a):
        self.test_nonincreasing(n_paths, risk, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(
        self, n_paths, risk, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(risk).to(device)
        x1 = torch.randn(n_paths).to(device)
        x2 = torch.randn(n_paths).to(device)
        assert_convex(loss, x1, x2, a)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex_gpu(self, n_paths, risk, a):
        self.test_convex(n_paths, risk, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_cash(self, n_paths, a, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(a).to(device)
        x = torch.randn(n_paths).to(device)
        assert_cash_equivalent(loss, x, float(loss.cash(x).item()))

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_cash_gpu(self, n_paths, a):
        self.test_cash(n_paths, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("c", [0.001, 1, 2])
    def test_cash_equivalent(
        self, n_paths, risk, c, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(risk).to(device)
        assert_cash_invariant(loss, torch.randn(n_paths), c)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("c", [0.001, 1, 2])
    def test_cash_equivalent_gpu(self, n_paths, risk, c):
        self.test_cash_equivalent(n_paths, risk, c, device="cuda")

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_value(
        self, n_paths, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        value = 1.0
        loss = EntropicRiskMeasure(a).to(device)
        result = loss(torch.full((n_paths,), value).to(device))
        expect = torch.log(torch.exp(-a * torch.tensor(value).to(device))) / a
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_value_gpu(self, n_paths, a):
        self.test_value(n_paths, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("value", [100.0, -100.0])
    def test_extreme(self, n_paths, a, value):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(a)
        result = loss(torch.full((n_paths,), value))
        assert_close(result, torch.tensor(-value))

    def test_extreme2(self):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(a=1.0)
        result1 = loss(torch.tensor([1000.0, 0.0]))
        result2 = loss(torch.tensor([500.0, -500.0]))
        result3 = loss(torch.tensor([0.0, -1000.0]))
        assert_close(result1 + 500, result2)
        assert_close(result1 + 1000, result3)

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("value", [100.0, -100.0])
    def test_extreme(self, n_paths, a, value):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(a)
        result = loss(torch.full((n_paths,), value))
        assert_close(result, torch.tensor(-value))

    def test_extreme2(self):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure(a=1.0)
        result1 = loss(torch.tensor([1000.0, 0.0]))
        result2 = loss(torch.tensor([500.0, -500.0]))
        result3 = loss(torch.tensor([0.0, -1000.0]))
        assert_close(result1 + 500, result2)
        assert_close(result1 + 1000, result3)

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

    def test_shape(self, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        loss = EntropicRiskMeasure().to(device)
        assert_loss_shape(loss, device=device)

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")


class TestEntropicLoss:
    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(
        self, n_paths, risk, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = EntropicLoss(risk).to(device)
        x1 = torch.randn(n_paths).to(device)
        x2 = x1 - a
        assert_monotone(loss, x1, x2, increasing=False)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 10.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing_gpu(self, n_paths, risk, a):
        self.test_nonincreasing(n_paths, risk, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(
        self, n_paths, risk, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = EntropicLoss(risk).to(device)
        input1 = torch.randn(n_paths).to(device)
        input2 = torch.randn(n_paths).to(device)
        assert_convex(loss, input1, input2, a)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [1.0, 2.0, 3.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex_gpu(self, n_paths, risk, a):
        self.test_convex(n_paths, risk, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_cash(self, n_paths, a, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        loss = EntropicLoss(a).to(device)
        x = torch.randn(n_paths).to(device)
        assert_cash_equivalent(loss, x, float(loss.cash(x).item()))

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_cash_gpu(self, n_paths, a):
        self.test_cash(n_paths, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_value(
        self, n_paths, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        value = 1.0
        loss = EntropicLoss(a).to(device)
        result = loss(torch.full((n_paths,), value).to(device))
        expect = torch.exp(-a * torch.tensor(value).to(device))
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("a", [1.0, 2.0, 3.0])
    def test_value_gpu(self, n_paths, a):
        self.test_value(n_paths, a, device="cuda")

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

    def test_shape(self, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        loss = EntropicLoss().to(device)
        assert_loss_shape(loss, device=device)

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")


class TestIsoelasticLoss:
    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(
        self, n_paths, risk, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = IsoelasticLoss(risk).to(device)
        x2 = torch.randn(n_paths).to(device).exp()  # force positive
        x1 = x2 + a

        assert_monotone(loss, x1, x2, increasing=False)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing_gpu(self, n_paths, risk, a):
        self.test_nonincreasing(n_paths, risk, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(
        self, n_paths, risk, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = IsoelasticLoss(risk).to(device)
        x1 = torch.randn(n_paths).to(device).exp()
        x2 = torch.randn(n_paths).to(device).exp()
        assert_convex(loss, x1, x2, a)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [1, 10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex_gpu(self, n_paths, risk, a):
        self.test_convex(n_paths, risk, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    def test_cash(
        self, n_paths, risk, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = IsoelasticLoss(risk).to(device)
        x = torch.randn(n_paths).to(device).exp()  # force positive
        assert_cash_equivalent(loss, x, float(loss.cash(x).item()))

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("risk", [0.1, 0.5, 1.0])
    def test_cash_gpu(self, n_paths, risk):
        self.test_cash(n_paths, risk, device="cuda")

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

    def test_shape(self, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        loss = IsoelasticLoss(0.5).to(device)
        assert_loss_shape(loss, device=device)

        loss = IsoelasticLoss(1.0).to(device)
        assert_loss_shape(loss, device=device)

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")


class TestExpectedShortFall:
    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.5])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing(
        self, n_paths, p, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = ExpectedShortfall(p).to(device)
        x1 = torch.randn(n_paths).to(device)
        x2 = x1 - 1
        assert_monotone(loss, x1, x2, increasing=False)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.5])
    @pytest.mark.parametrize("a", [0.001, 1, 2])
    def test_nonincreasing_gpu(self, n_paths, p, a):
        self.test_nonincreasing(n_paths, p, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex(
        self, n_paths, p, a, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        loss = ExpectedShortfall(p).to(device)
        x1 = torch.randn(n_paths).to(device)
        x2 = torch.randn(n_paths).to(device)
        assert_convex(loss, x1, x2, a)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("a", [0.1, 0.5])
    def test_convex_gpu(self, n_paths, p, a):
        self.test_convex(n_paths, p, a, device="cuda")

    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_cash(self, n_paths, p, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        loss = ExpectedShortfall(p).to(device)
        x = torch.randn(n_paths).to(device)
        assert_cash_equivalent(loss, x, float(loss.cash(x).item()))

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [100, 1000])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_cash_gpu(self, n_paths, p):
        self.test_cash(n_paths, p, device="cuda")

    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("eta", [0.001, 1, 2])
    def test_cash_equivalent(
        self, n_paths, p, eta, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        loss = ExpectedShortfall(p).to(device)
        assert_cash_invariant(loss, torch.randn(n_paths).to(device), eta)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [10, 100])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("eta", [0.001, 1, 2])
    def test_cash_equivalent_gpu(self, n_paths, p, eta):
        self.test_cash_equivalent(n_paths, p, eta, device="cuda")

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
    def test_value(
        self, percentile, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        torch.manual_seed(42)

        n_paths = 100
        k = int(n_paths * percentile)
        loss = ExpectedShortfall(percentile).to(device)

        input = torch.randn(n_paths).to(device)

        result = loss(input)
        expect = -torch.mean(torch.tensor(sorted(input)[:k]).to(device))

        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("percentile", [0.1, 0.5, 0.9])
    def test_value_gpu(self, percentile):
        self.test_value(percentile, device="cuda")

    def test_repr(self):
        loss = ExpectedShortfall(0.1)
        assert repr(loss) == "ExpectedShortfall(0.1)"
        loss = ExpectedShortfall(0.5)
        assert repr(loss) == "ExpectedShortfall(0.5)"

    def test_shape(self, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        loss = ExpectedShortfall().to(device)
        assert_loss_shape(loss, device=device)

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")


class TestOCE:
    def train_oce(self, m, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        optim = torch.optim.Adam(m.parameters())

        for _ in range(1000):
            optim.zero_grad()
            m(torch.randn(10000).to(device)).backward()
            optim.step()

    def test_fit(self, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        m = OCE(lambda input: 1 - torch.exp(-input)).to(device)

        self.train_oce(m, device=device)

        input = torch.randn(10000).to(device)

        result = m(input)
        expect = torch.log(EntropicLoss()(input))

        assert_close(result, expect, rtol=1e-02, atol=1e-5)

    @pytest.mark.gpu
    def test_fit_gpu(self):
        self.test_fit(device="cuda")

    def test_repr(self):
        def exp_utility(input):
            return 1 - torch.exp(-input)

        loss = OCE(exp_utility)
        assert repr(loss) == "OCE(exp_utility, w=0.)"

    def test_shape(self, device: Optional[Union[str, torch.device]] = "cpu"):
        torch.manual_seed(42)

        loss = OCE(lambda input: 1 - torch.exp(-input)).to(device)
        assert_loss_shape(loss, device=device)

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")
