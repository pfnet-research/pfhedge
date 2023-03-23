import pytest
import torch

from pfhedge import autogreek
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.instruments import LookbackOption
from pfhedge.nn import Hedger
from pfhedge.nn import WhalleyWilmott


class TestWhalleyWilmott:
    def test_repr(self):
        derivative = EuropeanOption(BrownianStock())
        m = WhalleyWilmott(derivative)
        expect = """\
WhalleyWilmott(
  (bs): BSEuropeanOption(strike=1.)
)"""
        assert repr(m) == expect

        derivative = EuropeanOption(BrownianStock())
        m = WhalleyWilmott(derivative, a=2.0)
        expect = """\
WhalleyWilmott(
  a=2.
  (bs): BSEuropeanOption(strike=1.)
)"""
        assert repr(m) == expect

        derivative = LookbackOption(BrownianStock())
        m = WhalleyWilmott(derivative)
        expect = """\
WhalleyWilmott(
  (bs): BSLookbackOption(strike=1.)
)"""
        assert repr(m) == expect

    def test_shape(self, device: str = "cpu"):
        torch.distributions.Distribution.set_default_validate_args(False)

        deriv = EuropeanOption(BrownianStock()).to(device)
        m = WhalleyWilmott(deriv).to(device)

        N = 10
        H_in = len(m.inputs())
        M_1 = 11
        M_2 = 12

        input = torch.zeros((N, H_in)).to(device)
        assert m(input).size() == torch.Size((N, 1))

        input = torch.zeros((N, M_1, H_in)).to(device)
        assert m(input).size() == torch.Size((N, M_1, 1))

        input = torch.zeros((N, M_1, M_2, H_in)).to(device)
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")

    def test(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(cost=1e-4)).to(device)
        model = WhalleyWilmott(derivative).to(device)
        hedger = Hedger(model, model.inputs())
        pnl = hedger.compute_pnl(derivative)
        assert not pnl.isnan().any()

    @pytest.mark.gpu
    def test_gpu(self):
        self.test(device="cuda")

    def test_autogreek_generate_nan_for_float64(self, device: str = "cpu"):
        derivative = (
            EuropeanOption(BrownianStock(cost=1e-4)).to(torch.float64).to(device)
        )
        model = WhalleyWilmott(derivative).to(torch.float64).to(device)
        hedger = Hedger(model, model.inputs()).to(torch.float64).to(device)

        def pricer(spot):
            return hedger.price(derivative, init_state=(spot,), enable_grad=True)

        delta = autogreek.delta(pricer, spot=torch.tensor(1.0).to(device))
        assert not delta.isnan().any()

    @pytest.mark.gpu
    def test_autogreek_generate_nan_for_float64_gpu(self):
        self.test_autogreek_generate_nan_for_float64(device="cuda")
