from unittest.mock import patch

import pytest
import torch
from torch import Tensor
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.testing import assert_close

from pfhedge.instruments import BaseDerivative
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


class ZeroDerivative(BaseDerivative):
    def __init__(self, underlier, maturity=20 / 250):
        super().__init__()
        self.underlier = underlier
        self.maturity = maturity

    def payoff_fn(self):
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
    def test_error_optimizer(self, device: str = "cpu"):
        hedger = Hedger(Linear(2, 1), ["moneyness", "time_to_maturity"]).to(device)
        derivative = EuropeanOption(BrownianStock()).to(device)
        with pytest.raises(TypeError):
            hedger._configure_optimizer(derivative, optimizer=Identity)

    @pytest.mark.gpu
    def test_error_optimizer_gpu(self):
        self.test_error_optimizer(device="cuda")

    def test_repr(self):
        hedger = Hedger(Linear(2, 1), ["moneyness", "time_to_maturity"])
        expect = """\
Hedger(
  inputs=['moneyness', 'time_to_maturity']
  (model): Linear(in_features=2, out_features=1, bias=True)
  (criterion): EntropicRiskMeasure()
)"""
        assert repr(hedger) == expect

        derivative = EuropeanOption(BrownianStock())
        model = BlackScholes(derivative)
        hedger = Hedger(model, model.inputs())
        expect = """\
Hedger(
  inputs=['log_moneyness', 'time_to_maturity', 'volatility']
  (model): BSEuropeanOption(strike=1.)
  (criterion): EntropicRiskMeasure()
)"""
        assert repr(hedger) == expect

    def test_compute_hedge_error_not_same_size(self, device: str = "cpu"):
        stock0 = BrownianStock().to(device)
        stock1 = BrownianStock().to(device)
        stock0.register_buffer("spot", torch.ones(2, 3).to(device))
        derivative = EuropeanOption(stock0).to(device)
        hedger = Hedger(Naked(), ["empty"]).to(device)

        stock1.register_buffer("spot", torch.ones(2, 4).to(device))
        with pytest.raises(ValueError):
            _ = hedger.compute_hedge(derivative, hedge=[stock0, stock1])
        stock1.register_buffer("spot", torch.ones(3, 3).to(device))
        with pytest.raises(ValueError):
            _ = hedger.compute_hedge(derivative, hedge=[stock0, stock1])

    @pytest.mark.gpu
    def test_compute_hedge_error_not_same_size_gpu(self):
        self.test_compute_hedge_error_not_same_size(device="cuda")

    @pytest.mark.parametrize("hin", [1, 2])
    def test_compute_pnl_size(self, hin, device: str = "cpu"):
        torch.manual_seed(42)

        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Linear(hin, 1), inputs=["empty"] * hin).to(device)
        pnl = hedger.compute_pnl(derivative)

        N = derivative.ul().spot.size(0)
        assert pnl.size() == torch.Size((N,))

        torch.manual_seed(42)
        derivative = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(Linear(hin + 1, 1), inputs=["empty"] * hin + ["prev_hedge"]).to(
            device
        )
        pnl = hedger.compute_pnl(derivative)

        N = derivative.ul().spot.size(0)
        assert pnl.size() == torch.Size((N,))

    @pytest.mark.gpu
    @pytest.mark.parametrize("hin", [1, 2])
    def test_compute_pnl_size_gpu(self, hin):
        self.test_compute_pnl_size(hin=hin, device="cuda")

    @pytest.mark.parametrize("cost", [0.0, 1e-3])
    def test_compute_pnl_1(self, cost, device: str = "cpu"):
        # pnl = -payoff if output = 0
        torch.manual_seed(42)

        deriv = EuropeanOption(BrownianStock(cost=cost)).to(device)
        hedger = Hedger(Naked(), ["empty"]).to(device)

        pnl = hedger.compute_pnl(deriv)
        payoff = deriv.payoff()
        assert_close(pnl, -payoff)

        result = hedger.compute_pnl(deriv)
        expect = -deriv.payoff()
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("cost", [0.0, 1e-3])
    def test_compute_pnl_1_gpu(self, cost):
        self.test_compute_pnl_1(cost=cost, device="cuda")

    def test_compute_pnl_2(self, device: str = "cpu"):
        torch.manual_seed(42)
        N, T = 2, 4
        derivative = ZeroDerivative(BrownianStock()).to(device)
        output = torch.randn(N, T - 1).to(device)

        m = FakeModule(output.unsqueeze(-1)).to(device)
        hedger = Hedger(m, ["empty", "prev_hedge"]).to(device)

        spot = torch.randn(N, T).to(device).exp()
        derivative.ul().register_buffer("spot", spot)
        with patch("pfhedge.instruments.BrownianStock.simulate", void):
            # so that the simulation is not performed
            result = hedger.compute_pnl(derivative)

        expect = (spot.diff(dim=-1) * output).sum(-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_compute_pnl_2_gpu(self):
        self.test_compute_pnl_2(device="cuda")

    def test_compute_pnl_2_multiple_hedges(self, device: str = "cpu"):
        torch.manual_seed(42)
        N, T = 2, 4

        stock0 = BrownianStock().to(device)
        stock1 = BrownianStock().to(device)
        stock0.register_buffer("spot", torch.randn(N, T).to(device).exp())
        stock1.register_buffer("spot", torch.randn(N, T).to(device).exp())
        derivative = ZeroDerivative(stock0).to(device)

        output = torch.randn(N, T - 1, 2).to(device)
        m = FakeModule(output).to(device)
        hedger = Hedger(m, ["empty", "prev_hedge"])

        with patch("pfhedge.instruments.BrownianStock.simulate", void):
            # so that the simulation is not performed
            result = hedger.compute_pnl(derivative, hedge=[stock0, stock1])

        pnl0 = (stock0.spot.diff(dim=-1) * output[..., 0]).sum(-1)
        pnl1 = (stock1.spot.diff(dim=-1) * output[..., 1]).sum(-1)
        expect = pnl0 + pnl1
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_compute_pnl_2_multiple_hedges_gpu(self):
        self.test_compute_pnl_2_multiple_hedges(device="cuda")

    def test_compute_pnl_2_multiple_hedges_payoff(self, device: str = "cpu"):
        torch.manual_seed(42)

        N, T = 2, 4

        stock0 = BrownianStock().to(device)
        stock1 = BrownianStock().to(device)
        stock0.register_buffer("spot", torch.randn(N, T).to(device).exp())
        stock1.register_buffer("spot", torch.randn(N, T).to(device).exp())
        derivative = EuropeanOption(stock0).to(device)
        payoff = derivative.payoff()

        output = torch.randn(N, T - 1, 2).to(device)
        m = FakeModule(output).to(device)
        hedger = Hedger(m, ["empty", "prev_hedge"]).to(device)
        with patch("pfhedge.instruments.BrownianStock.simulate", void):
            result = hedger.compute_pnl(derivative, hedge=[stock0, stock1])
        pnl0 = (stock0.spot.diff(dim=-1) * output[..., 0]).sum(-1)
        pnl1 = (stock1.spot.diff(dim=-1) * output[..., 1]).sum(-1)
        expect = pnl0 + pnl1 - payoff
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_compute_pnl_2_multiple_hedges_payoff_gpu(self):
        self.test_compute_pnl_2_multiple_hedges_payoff(device="cuda")

    def test_compute_pnl_payoff(self, device: str = "cpu"):
        torch.manual_seed(42)

        N, T = 10, 20
        derivative0 = EuropeanOption(BrownianStock()).to(device)
        derivative1 = ZeroDerivative(BrownianStock()).to(device)
        output = torch.randn(N, T - 1)

        hedger0 = Hedger(FakeModule(output.unsqueeze(-1)), ["empty", "prev_hedge"]).to(
            device
        )
        hedger1 = Hedger(FakeModule(output.unsqueeze(-1)), ["empty", "prev_hedge"]).to(
            device
        )

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

    @pytest.mark.gpu
    def test_compute_pnl_payoff_gpu(self):
        self.test_compute_pnl_payoff(device="cuda")

    def test_compute_pnl_cost(self, device: str = "cpu"):
        cost = 1e-3
        N, T = 10, 20
        derivative0 = EuropeanOption(BrownianStock(cost=0.0)).to(device)
        derivative1 = EuropeanOption(BrownianStock(cost=cost)).to(device)
        output = torch.randn(N, T - 1).to(device)

        hedger0 = Hedger(FakeModule(output.unsqueeze(-1)), ["empty", "prev_hedge"]).to(
            device
        )
        hedger1 = Hedger(FakeModule(output.unsqueeze(-1)), ["empty", "prev_hedge"]).to(
            device
        )

        spot = torch.randn(N, T).to(device).exp()
        derivative0.ul().register_buffer("spot", spot)
        derivative1.ul().register_buffer("spot", spot)
        with patch("pfhedge.instruments.BrownianStock.simulate", void):
            # so that the simulation is not performed
            pnl0 = hedger0.compute_pnl(derivative0)
            pnl1 = hedger1.compute_pnl(derivative1)

        result = pnl0 - pnl1
        output = torch.cat((torch.zeros(N, 1).to(device), output), dim=-1)
        expect = cost * (spot[..., :-1] * output.diff(dim=-1).abs()).sum(-1)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_compute_pnl_cost_gpu(self):
        self.test_compute_pnl_cost(device="cuda")

    def test_forward_shape(self, device: str = "cpu"):
        torch.distributions.Distribution.set_default_validate_args(False)

        deriv = EuropeanOption(BrownianStock()).to(device)

        N = 2
        M_1 = 5
        M_2 = 6
        H_in = 3

        input = torch.zeros((N, M_1, M_2, H_in)).to(device)
        m = Hedger(MultiLayerPerceptron(), ["empty"]).to(device)
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

        model = BlackScholes(deriv).to(device)
        m = Hedger(model, model.inputs()).to(device)
        input = torch.zeros((N, M_1, M_2, len(model.inputs())))
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

        model = WhalleyWilmott(deriv).to(device)
        m = Hedger(model, model.inputs()).to(device)
        input = torch.zeros((N, M_1, M_2, len(model.inputs())))
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

        model = Naked().to(device)
        m = Hedger(model, ["empty"]).to(device)
        input = torch.zeros((N, M_1, M_2, 10)).to(device)
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))

    @pytest.mark.gpu
    def test_forward_shape_gpu(self):
        self.test_forward_shape(device="cuda")

    @pytest.mark.parametrize("h_in", [1, 2, 3])
    def test_get_input(self, h_in, device: str = "cpu"):
        hedger = Hedger(Naked(), ["zeros"] * h_in).to(device)
        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.simulate()
        _ = hedger.compute_pnl(derivative, n_paths=1)
        N, T = derivative.ul().spot.size()
        input = hedger.get_input(derivative, 0)
        assert input.size() == torch.Size((N, 1, h_in))
        input = hedger.get_input(derivative, None)
        assert input.size() == torch.Size((N, T, h_in))

    @pytest.mark.gpu
    @pytest.mark.parametrize("h_in", [1, 2, 3])
    def test_get_input_gpu(self, h_in):
        self.test_get_input(h_in=h_in, device="cuda")

    @pytest.mark.parametrize("h_in", [1, 2, 3])
    def test_compute_hedge(self, h_in, device: str = "cpu"):
        # test size
        hedger = Hedger(Naked(), ["zeros"] * h_in).to(device)
        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.simulate()
        hedge = hedger.compute_hedge(derivative)
        N, T = derivative.ul().spot.size()
        H = 1
        assert hedge.size() == torch.Size((N, H, T))

        # test size
        hedger = Hedger(Naked(), ["zeros"] * h_in + ["prev_hedge"]).to(device)
        derivative = EuropeanOption(BrownianStock()).to(device)
        derivative.simulate()
        hedge = hedger.compute_hedge(derivative)
        N, T = derivative.ul().spot.size()
        H = 1
        assert hedge.size() == torch.Size((N, H, T))

    @pytest.mark.gpu
    @pytest.mark.parametrize("h_in", [1, 2, 3])
    def test_compute_hedge_gpu(self, h_in):
        self.test_compute_hedge(h_in=h_in, device="cuda")

    def test_compute_loss(self, device: str = "cpu"):
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            Naked(), ["log_moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        result = hedger.compute_loss(deriv)
        expect = EntropicRiskMeasure().to(device)(-deriv.payoff())
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_compute_loss_gpu(self):
        self.test_compute_loss(device="cuda")

    def test_hedging_with_identical_derivative(self, device: str = "cpu"):
        torch.manual_seed(42)

        class Ones(Module):
            def forward(self, input: Tensor):
                return torch.ones_like(input[..., :1])

        def pricer(derivative: Derivative) -> Tensor:
            return (
                BlackScholes(derivative)
                .to(device)
                .price(
                    log_moneyness=derivative.log_moneyness(),
                    time_to_maturity=derivative.time_to_maturity(),
                    volatility=derivative.ul().volatility,
                )
            )

        derivative = EuropeanOption(BrownianStock(), maturity=5 / 250).to(device)
        derivative.list(pricer)
        hedger = Hedger(Ones(), ["empty", "prev_hedge"]).to(device)

        torch.manual_seed(42)
        result = hedger.compute_pnl(derivative, hedge=[derivative], n_paths=2)
        # value of a short position of the derivative
        expect = -derivative.spot[:, 0]
        assert_close(result, expect, check_stride=False)

        torch.manual_seed(42)
        result = hedger.price(derivative, hedge=[derivative], n_paths=2)
        expect = derivative.spot[0, 0]
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    def test_hedging_with_identical_derivative_gpu(self):
        self.test_hedging_with_identical_derivative(device="cuda")
