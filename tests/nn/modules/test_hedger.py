import pytest
import torch
from torch.nn import Identity
from torch.nn import Linear
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import EntropicRiskMeasure
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.nn import Naked
from pfhedge.nn import WhalleyWilmott


class TestHedger:
    """
    pfhedge.Hedger
    """

    def test_error_optimizer(self):
        hedger = Hedger(Linear(2, 1), ["moneyness", "time_to_maturity"])
        derivative = EuropeanOption(BrownianStock())
        with pytest.raises(TypeError):
            hedger.fit(derivative, optimizer=Identity)

    def test_repr(self):
        hedger = Hedger(Linear(2, 1), ["moneyness", "time_to_maturity"])
        assert repr(hedger) == (
            "Hedger(\n"
            "  inputs=['moneyness', 'time_to_maturity']\n"
            "  (model): Linear(in_features=2, out_features=1, bias=True)\n"
            "  (criterion): EntropicRiskMeasure()\n"
            ")"
        )

        derivative = EuropeanOption(BrownianStock())
        model = BlackScholes(derivative)
        hedger = Hedger(model, model.inputs())
        assert repr(hedger) == (
            "Hedger(\n"
            "  inputs=['log_moneyness', 'time_to_maturity', 'volatility']\n"
            "  (model): BSEuropeanOption()\n"
            "  (criterion): EntropicRiskMeasure()\n"
            ")"
        )

    def test_compute_pnl(self):
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock())
        hedger = Hedger(Naked(), ["empty"])

        pnl = hedger.compute_pnl(deriv)
        payoff = deriv.payoff()
        assert_close(pnl, -payoff)

        result = hedger.compute_pnl(deriv)
        expect = -deriv.payoff()
        assert_close(result, expect)

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        deriv = EuropeanOption(BrownianStock())

        N = 2
        M_1 = 5
        M_2 = 6
        H_in = 3

        input = torch.empty((N, M_1, M_2, H_in))
        m = Hedger(MultiLayerPerceptron(), ["empty"])
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

        model = BlackScholes(deriv)
        m = Hedger(model, model.inputs())
        input = torch.empty((N, M_1, M_2, len(model.inputs())))
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

        model = WhalleyWilmott(deriv)
        m = Hedger(model, model.inputs())
        input = torch.empty((N, M_1, M_2, len(model.inputs())))
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

        model = Naked()
        m = Hedger(model, ["empty"])
        input = torch.empty((N, M_1, M_2, 10))
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

    def test_compute_loss(self):
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock())
        hedger = Hedger(Naked(), ["log_moneyness", "time_to_maturity", "volatility"])

        result = hedger.compute_loss(deriv)
        expect = EntropicRiskMeasure()(-deriv.payoff())
        assert_close(result, expect)
