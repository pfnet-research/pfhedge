from unittest.mock import patch

import pytest
import torch
from torch import Tensor
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import Derivative
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import EntropicRiskMeasure
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.nn import Naked
from pfhedge.nn import WhalleyWilmott


def void(*args, **kwargs):
    pass


class ZeroDerivative(Derivative):
    def __init__(self, underlier, maturity=20 / 250):
        super().__init__()
        self.underlier = underlier
        self.maturity = maturity

    def payoff(self):
        return torch.zeros_like(self.ul().spot[..., 0])


class FakeModule(Module):
    def __init__(self, output: Tensor):
        # output: shape (N, T)
        super().__init__()
        self.i = 0
        self.register_buffer("output", output)

    def forward(self, input: Tensor):
        # output: shape (N, 1)
        output = self.get_buffer("output")[..., [self.i]]
        self.i += 1
        return output


class TestHedger:
    def test_fit_error_optimizer(self):
        hedger = Hedger(Linear(2, 1), ["moneyness", "expiry_time"])
        derivative = EuropeanOption(BrownianStock())
        with pytest.raises(TypeError):
            hedger.fit(derivative, optimizer=Identity)

    def test_repr(self):
        hedger = Hedger(Linear(2, 1), ["moneyness", "expiry_time"])
        assert repr(hedger) == (
            "Hedger(\n"
            "  inputs=['moneyness', 'expiry_time']\n"
            "  (model): Linear(in_features=2, out_features=1, bias=True)\n"
            "  (criterion): EntropicRiskMeasure()\n"
            ")"
        )

        derivative = EuropeanOption(BrownianStock())
        model = BlackScholes(derivative)
        hedger = Hedger(model, model.inputs())
        assert repr(hedger) == (
            "Hedger(\n"
            "  inputs=['log_moneyness', 'expiry_time', 'volatility']\n"
            "  (model): BSEuropeanOption(strike=1.)\n"
            "  (criterion): EntropicRiskMeasure()\n"
            ")"
        )

    @pytest.mark.parametrize("cost", [0.0, 1e-3])
    def test_compute_pnl_1(self, cost):
        # pnl = -payoff if output = 0
        torch.manual_seed(42)

        deriv = EuropeanOption(BrownianStock(cost=cost))
        hedger = Hedger(Naked(), ["empty"])

        pnl = hedger.compute_pnl(deriv)
        payoff = deriv.payoff()
        assert_close(pnl, -payoff)

        result = hedger.compute_pnl(deriv)
        expect = -deriv.payoff()
        assert_close(result, expect)

    def test_compute_pnl_2(self):
        torch.manual_seed(42)
        N, T = 10, 20

        derivative = ZeroDerivative(BrownianStock())
        output = torch.randn(N, T - 1)

        m = FakeModule(output)
        hedger = Hedger(m, ["empty"])

        spot = torch.randn(N, T).exp()
        derivative.ul().register_buffer("spot", spot)
        with patch("pfhedge.instruments.BrownianStock.simulate", void):
            # so that the simulation is not performed
            result = hedger.compute_pnl(derivative)

        expect = (spot.diff(dim=-1) * output).sum(-1)
        assert_close(result, expect)

    def test_compute_pnl_payoff(self):
        N, T = 10, 20
        derivative0 = EuropeanOption(BrownianStock())
        derivative1 = ZeroDerivative(BrownianStock())
        output = torch.randn(N, T - 1)

        hedger0 = Hedger(FakeModule(output), ["empty"])
        hedger1 = Hedger(FakeModule(output), ["empty"])

        spot = torch.randn(N, T).exp()
        derivative0.ul().register_buffer("spot", spot)
        derivative1.ul().register_buffer("spot", spot)
        with patch("pfhedge.instruments.BrownianStock.simulate", void):
            # so that the simulation is not performed
            pnl0 = hedger0.compute_pnl(derivative0)
            pnl1 = hedger1.compute_pnl(derivative1)

        result = pnl0 - pnl1
        expect = -derivative0.payoff()
        assert_close(result, expect)

    def test_compute_pnl_cost(self):
        cost = 1e-3
        N, T = 10, 20
        derivative0 = EuropeanOption(BrownianStock(cost=0.0))
        derivative1 = EuropeanOption(BrownianStock(cost=cost))
        output = torch.randn(N, T - 1)

        hedger0 = Hedger(FakeModule(output), ["empty"])
        hedger1 = Hedger(FakeModule(output), ["empty"])

        spot = torch.randn(N, T).exp()
        derivative0.ul().register_buffer("spot", spot)
        derivative1.ul().register_buffer("spot", spot)
        with patch("pfhedge.instruments.BrownianStock.simulate", void):
            # so that the simulation is not performed
            pnl0 = hedger0.compute_pnl(derivative0)
            pnl1 = hedger1.compute_pnl(derivative1)

        result = pnl0 - pnl1
        output = torch.cat((torch.zeros(N, 1), output), dim=-1)
        expect = cost * (spot[..., :-1] * output.diff(dim=-1).abs()).sum(-1)
        assert_close(result, expect)

    def test_forward_shape(self):
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
        hedger = Hedger(Naked(), ["log_moneyness", "expiry_time", "volatility"])

        result = hedger.compute_loss(deriv)
        expect = EntropicRiskMeasure()(-deriv.payoff())
        assert_close(result, expect)

    def test_hedging_with_identical_derivative(self):
        torch.manual_seed(42)

        class Ones(Module):
            def forward(self, input: Tensor):
                return torch.ones_like(input[:, :1])

        pricer = lambda derivative: BlackScholes(derivative).price(
            log_moneyness=derivative.log_moneyness(),
            expiry_time=derivative.time_to_maturity(),
            volatility=derivative.ul().volatility,
        )

        derivative = EuropeanOption(BrownianStock(), maturity=5 / 250)
        derivative.list(pricer)
        hedger = Hedger(Ones(), ["empty"])

        torch.manual_seed(42)
        result = hedger.compute_pnl(derivative, hedge=derivative, n_paths=2)
        # value of a short position of the derivative
        expect = -derivative.spot[:, 0]
        assert_close(result, expect, check_stride=False)

        torch.manual_seed(42)
        result = hedger.price(derivative, hedge=derivative, n_paths=2)
        expect = derivative.spot[0, 0]
        assert_close(result, expect, check_stride=False)
