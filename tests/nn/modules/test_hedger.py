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
        # output: shape (N, T, *)
        super().__init__()
        self.i = 0
        self.register_buffer("output", output)

    def forward(self, input: Tensor):
        # output: shape (N, 1, *)
        output = self.get_buffer("output")[:, [self.i]]
        self.i += 1
        return output


class TestHedger:
    def test_fit_error_optimizer(self):
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
            "  (model): BSEuropeanOption(strike=1.)\n"
            "  (criterion): EntropicRiskMeasure()\n"
            ")"
        )

    @pytest.mark.parametrize("hin", [1, 2])
    def test_compute_pnl_size(self, hin):
        torch.manual_seed(42)

        derivative = EuropeanOption(BrownianStock())
        hedger = Hedger(Linear(hin, 1), inputs=["empty"] * hin)
        pnl = hedger.compute_pnl(derivative)

        N = derivative.ul().spot.size(0)
        assert pnl.size() == torch.Size((N,))

        torch.manual_seed(42)
        derivative = EuropeanOption(BrownianStock())
        hedger = Hedger(Linear(hin + 1, 1), inputs=["empty"] * hin + ["prev_hedge"])
        pnl = hedger.compute_pnl(derivative)

        N = derivative.ul().spot.size(0)
        assert pnl.size() == torch.Size((N,))

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
        N, T = 2, 4
        derivative = ZeroDerivative(BrownianStock())
        output = torch.randn(N, T - 1)

        m = FakeModule(output.unsqueeze(-1))
        hedger = Hedger(m, ["empty", "prev_hedge"])

        spot = torch.randn(N, T).exp()
        derivative.ul().register_buffer("spot", spot)
        with patch("pfhedge.instruments.BrownianStock.simulate", void):
            # so that the simulation is not performed
            result = hedger.compute_pnl(derivative)

        expect = (spot.diff(dim=-1) * output).sum(-1)
        print("spot\n", spot)
        print("spot.diff\n", spot.diff(dim=-1))
        print("output\n", output)
        print("result\n", result)
        print("expect\n", expect)
        assert_close(result, expect)

    def test_compute_pnl_payoff(self):
        torch.manual_seed(42)

        N, T = 10, 20
        derivative0 = EuropeanOption(BrownianStock())
        derivative1 = ZeroDerivative(BrownianStock())
        output = torch.randn(N, T - 1)

        hedger0 = Hedger(FakeModule(output.unsqueeze(-1)), ["empty", "prev_hedge"])
        hedger1 = Hedger(FakeModule(output.unsqueeze(-1)), ["empty", "prev_hedge"])

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

        hedger0 = Hedger(FakeModule(output.unsqueeze(-1)), ["empty", "prev_hedge"])
        hedger1 = Hedger(FakeModule(output.unsqueeze(-1)), ["empty", "prev_hedge"])

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

    @pytest.mark.parametrize("h_in", [1, 2, 3])
    def test_get_input(self, h_in):
        hedger = Hedger(Naked(), ["zeros"] * h_in)
        derivative = EuropeanOption(BrownianStock())
        derivative.simulate()
        hedger.inputs = [f.of(derivative, self) for f in hedger.inputs]
        N, T = derivative.ul().spot.size()
        input = hedger.get_input(0)
        assert input.size() == torch.Size((N, 1, h_in))
        input = hedger.get_input(None)
        assert input.size() == torch.Size((N, T, h_in))

    @pytest.mark.parametrize("h_in", [1, 2, 3])
    def test_compute_hedge(self, h_in):
        # test size
        hedger = Hedger(Naked(), ["zeros"] * h_in)
        derivative = EuropeanOption(BrownianStock())
        derivative.simulate()
        hedge = hedger.compute_hedge(derivative)
        N, T = derivative.ul().spot.size()
        H = 1
        assert hedge.size() == torch.Size((N, H, T))

        # test size
        hedger = Hedger(Naked(), ["zeros"] * h_in + ["prev_hedge"])
        derivative = EuropeanOption(BrownianStock())
        derivative.simulate()
        hedge = hedger.compute_hedge(derivative)
        N, T = derivative.ul().spot.size()
        H = 1
        assert hedge.size() == torch.Size((N, H, T))

    def test_compute_loss(self):
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock())
        hedger = Hedger(Naked(), ["log_moneyness", "time_to_maturity", "volatility"])

        result = hedger.compute_loss(deriv)
        expect = EntropicRiskMeasure()(-deriv.payoff())
        assert_close(result, expect)

    def test_hedging_with_identical_derivative(self):
        torch.manual_seed(42)

        class Ones(Module):
            def forward(self, input: Tensor):
                return torch.ones_like(input[..., :1])

        pricer = lambda derivative: BlackScholes(derivative).price(
            log_moneyness=derivative.log_moneyness(),
            time_to_maturity=derivative.time_to_maturity(),
            volatility=derivative.ul().volatility,
        )

        derivative = EuropeanOption(BrownianStock(), maturity=5 / 250)
        derivative.list(pricer)
        hedger = Hedger(Ones(), ["empty", "prev_hedge"])

        torch.manual_seed(42)
        result = hedger.compute_pnl(derivative, hedge=derivative, n_paths=2)
        # value of a short position of the derivative
        expect = -derivative.spot[:, 0]
        print("spot\n", derivative.spot)
        print("spot.diff\n", derivative.spot.diff(dim=-1))
        print("spot.diff.sum\n", derivative.spot.diff(dim=-1).sum(-1))
        print("payoff\n", derivative.payoff())
        print("result\n", result)
        print("expect\n", expect)
        assert_close(result, expect, check_stride=False)

        torch.manual_seed(42)
        result = hedger.price(derivative, hedge=derivative, n_paths=2)
        expect = derivative.spot[0, 0]
        assert_close(result, expect, check_stride=False)
