import torch
from pfhedge.nn import Hedger

from pfhedge.instruments import BrownianStock
from pfhedge import autogreek
from pfhedge.instruments import EuropeanOption
from pfhedge.instruments import LookbackOption
from pfhedge.nn import WhalleyWilmott


class TestWhalleyWilmott:
    def test_repr(self):
        derivative = EuropeanOption(BrownianStock())
        m = WhalleyWilmott(derivative)
        assert repr(m) == (
            "WhalleyWilmott(\n"
            "  (bs): BSEuropeanOption()\n"
            "  (clamp): Clamp()\n"
            ")"
        )

        derivative = EuropeanOption(BrownianStock())
        m = WhalleyWilmott(derivative, a=2)
        assert repr(m) == (
            "WhalleyWilmott(\n"
            "  a=2\n"
            "  (bs): BSEuropeanOption()\n"
            "  (clamp): Clamp()\n"
            ")"
        )

        derivative = LookbackOption(BrownianStock())
        m = WhalleyWilmott(derivative)
        assert repr(m) == (
            "WhalleyWilmott(\n"
            "  (bs): BSLookbackOption()\n"
            "  (clamp): Clamp()\n"
            ")"
        )

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        deriv = EuropeanOption(BrownianStock())
        m = WhalleyWilmott(deriv)

        N = 10
        H_in = len(m.inputs())
        M_1 = 11
        M_2 = 12

        input = torch.empty((N, H_in))
        assert m(input).size() == torch.Size((N, 1))

        input = torch.empty((N, M_1, H_in))
        assert m(input).size() == torch.Size((N, M_1, 1))

        input = torch.empty((N, M_1, M_2, H_in))
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

    def test(self):
        derivative = EuropeanOption(BrownianStock(cost=1e-4))
        model = WhalleyWilmott(derivative)
        hedger = Hedger(model, model.inputs())
        pnl = hedger.compute_pnl(derivative)
        assert not pnl.isnan().any()

    def test_autogreek(self):
        with torch.autograd.detect_anomaly():
            derivative = EuropeanOption(BrownianStock(cost=1e-4)).to(torch.float64)
            model = WhalleyWilmott(derivative).to(torch.float64)
            hedger = Hedger(model, model.inputs()).to(torch.float64)

            def pricer(spot):
                return hedger.price(
                derivative, init_state=(spot,), enable_grad=True
                )

            delta = autogreek.delta(pricer, spot=torch.tensor(1.0))
            assert not delta.isnan().any()
