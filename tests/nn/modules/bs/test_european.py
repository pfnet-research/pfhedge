import pytest
import torch
from torch import Tensor
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BSEuropeanOption

from ._base import _TestBSModule
from ._utils import compute_delta
from ._utils import compute_gamma
from ._utils import compute_price


class TestBSEuropeanOption(_TestBSModule):
    def test_repr(self):
        m = BSEuropeanOption()
        assert repr(m) == "BSEuropeanOption(strike=1.)"

        derivative = EuropeanOption(BrownianStock(), strike=1.1, call=False)
        m = BSEuropeanOption.from_derivative(derivative)
        assert repr(m) == "BSEuropeanOption(call=False, strike=1.1000)"

    def test_features(self):
        m = BSEuropeanOption()
        assert m.inputs() == ["log_moneyness", "time_to_maturity", "volatility"]
        _ = [get_feature(f) for f in m.inputs()]

    def test_delta_limit(self):
        EPSILON = 1e-10
        c = BSEuropeanOption()
        p = BSEuropeanOption(call=False)

        # delta = 0 (call), -1 (put) for spot --> +0
        result = compute_delta(c, torch.tensor([[-10.0, 1.0, 0.2]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_delta(p, torch.tensor([[-10.0, 1.0, 0.2]]))
        assert_close(result, torch.tensor([-1.0]))

        # delta = 1 (call), 0 (put) for spot --> +inf
        result = compute_delta(c, torch.tensor([[10.0, 1.0, 0.2]]))
        assert_close(result, torch.tensor([1.0]))
        result = compute_delta(p, torch.tensor([[10.0, 1.0, 0.2]]))
        assert_close(result, torch.tensor([0.0]))

        # delta = 0 (call), -1 (put) for spot < k and time --> +0
        result = compute_delta(c, torch.tensor([[-0.01, EPSILON, 0.2]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_delta(p, torch.tensor([[-0.01, EPSILON, 0.2]]))
        assert_close(result, torch.tensor([-1.0]))

        # delta = 1 (call), 0 (put) for spot > k and time --> +0
        result = compute_delta(c, torch.tensor([[0.01, EPSILON, 0.2]]))
        assert_close(result, torch.tensor([1.0]))
        result = compute_delta(p, torch.tensor([[0.01, EPSILON, 0.2]]))
        assert_close(result, torch.tensor([0.0]))

        # delta = 0 (call), -1 (put) for spot < k and volatility --> +0
        result = compute_delta(c, torch.tensor([[-0.01, 1.0, EPSILON]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_delta(p, torch.tensor([[-0.01, 1.0, EPSILON]]))
        assert_close(result, torch.tensor([-1.0]))

        # delta = 1 (call), 0 (put) for spot > k and volatility --> +0
        result = compute_delta(c, torch.tensor([[0.01, 1.0, EPSILON]]))
        assert_close(result, torch.tensor([1.0]))
        result = compute_delta(p, torch.tensor([[0.01, 1.0, EPSILON]]))
        assert_close(result, torch.tensor([0.0]))

    def test_gamma_limit(self):
        EPSILON = 1e-10
        c = BSEuropeanOption()
        p = BSEuropeanOption(call=False)

        # gamma = 0 for spot --> +0
        result = compute_gamma(c, torch.tensor([[-10.0, 1.0, 0.2]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_gamma(p, torch.tensor([[-10.0, 1.0, 0.2]]))
        assert_close(result, torch.tensor([0.0]))

        # gamma = 0 for spot --> +inf
        result = compute_gamma(c, torch.tensor([[10.0, 1.0, 0.2]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_gamma(p, torch.tensor([[10.0, 1.0, 0.2]]))
        assert_close(result, torch.tensor([0.0]))

        # gamma = 0 for spot / k < 1 and time --> +0
        result = compute_gamma(c, torch.tensor([[-0.01, EPSILON, 0.2]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_gamma(p, torch.tensor([[-0.01, EPSILON, 0.2]]))
        assert_close(result, torch.tensor([0.0]))

        # gamma = 0 for spot / k > 1 and time --> +0
        result = compute_gamma(c, torch.tensor([[0.01, EPSILON, 0.2]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_gamma(p, torch.tensor([[0.01, EPSILON, 0.2]]))
        assert_close(result, torch.tensor([0.0]))

        # gamma = 0 for spot / k < 1 and volatility --> +0
        result = compute_gamma(c, torch.tensor([[-0.01, 1.0, EPSILON]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_gamma(p, torch.tensor([[-0.01, 1.0, EPSILON]]))
        assert_close(result, torch.tensor([0.0]))

        # gamma = 0 for spot / k > 1 and volatility --> +0
        result = compute_gamma(c, torch.tensor([[0.01, 1.0, EPSILON]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_gamma(p, torch.tensor([[0.01, 1.0, EPSILON]]))
        assert_close(result, torch.tensor([0.0]))

    def test_price_limit(self):
        EPSILON = 1e-10
        c = BSEuropeanOption()
        p = BSEuropeanOption(call=False)

        # price = 0 (call), k - spot (put) for spot --> +0
        s = torch.tensor([-10.0]).exp()
        result = compute_price(c, torch.tensor([[s.log(), 1.0, 0.2]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_price(p, torch.tensor([[s.log(), 1.0, 0.2]]))
        assert_close(result, torch.tensor([1.0 - s]))

        # price = spot - k (call), 0 (put) for spot --> +inf
        s = torch.tensor([10.0]).exp()
        result = compute_price(c, torch.tensor([[s.log(), 1.0, 0.2]]))
        assert_close(result, torch.tensor([s - 1.0]))
        result = compute_price(p, torch.tensor([[s.log(), 1.0, 0.2]]))
        assert_close(result, torch.tensor([0.0]))

        # price = 0 (call), k - s (put) for spot < k and time --> +0
        s = torch.tensor([-0.01]).exp()
        result = compute_price(c, torch.tensor([[s.log(), EPSILON, 0.2]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_price(p, torch.tensor([[s.log(), EPSILON, 0.2]]))
        assert_close(result, torch.tensor([1.0 - s]))

        # price = spot - k (call), 0 (put) for spot > k and time --> +0
        s = torch.tensor([0.01]).exp()
        result = compute_price(c, torch.tensor([[s.log(), EPSILON, 0.2]]))
        assert_close(result, torch.tensor([s - 1.0]))
        result = compute_price(p, torch.tensor([[s.log(), EPSILON, 0.2]]))
        assert_close(result, torch.tensor([0.0]))

        # price = 0 (call), k - spot (put) for spot < k and volatility --> +0
        s = torch.tensor([-0.01]).exp()
        result = compute_price(c, torch.tensor([[s.log(), 1.0, EPSILON]]))
        assert_close(result, torch.tensor([0.0]))
        result = compute_price(p, torch.tensor([[s.log(), 1.0, EPSILON]]))
        assert_close(result, torch.tensor([1.0 - s]))

        # price = spot - k (call), 0 (put) for spot > k and volatility --> +0
        s = torch.tensor([0.01]).exp()
        result = compute_price(c, torch.tensor([[s.log(), 1.0, EPSILON]]))
        assert_close(result, torch.tensor([s - 1.0]))
        result = compute_price(p, torch.tensor([[s.log(), 1.0, EPSILON]]))
        assert_close(result, torch.tensor([0.0]))

    def test_price_monte_carlo(self):
        d = EuropeanOption(BrownianStock())
        m = BSEuropeanOption.from_derivative(d)
        torch.manual_seed(42)
        d.simulate(n_paths=int(1e6))

        input = torch.tensor([[0.0, d.maturity, d.ul().sigma]])
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        print(result, expect)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

        d = EuropeanOption(BrownianStock(), call=False)
        m = BSEuropeanOption.from_derivative(d)
        torch.manual_seed(42)
        d.simulate(n_paths=int(1e6))

        input = torch.tensor([[0.0, d.maturity, d.ul().sigma]])
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        print(result, expect)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

    def test_forward_2(self):
        m = BSEuropeanOption(call=False)
        input = torch.tensor([[0.0, 1.0, 0.2]])
        result = m(input)
        expect = torch.full_like(result, -0.4601721)
        assert_close(result, expect)

    def test_forward_3(self):
        derivative = EuropeanOption(BrownianStock(), call=False)
        m = BSEuropeanOption.from_derivative(derivative)
        input = torch.tensor([[0.0, 1.0, 0.2]])
        result = m(input)
        expect = torch.full_like(result, -0.4601721)
        assert_close(result, expect)

    def test_delta_1(self):
        m = BSEuropeanOption()
        result = m.delta(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.2))
        expect = torch.full_like(result, 0.5398278962)
        assert_close(result, expect)

    def test_delta_2(self):
        m = BSEuropeanOption(call=False)
        result = m.delta(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.2))
        expect = torch.full_like(result, -0.4601721)
        assert_close(result, expect)

    @pytest.mark.parametrize("call", [True, False])
    def test_delta_3(self, call: bool):
        m = BSEuropeanOption(call=call)
        with pytest.raises(ValueError):
            m.delta(torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(0.2))
        with pytest.raises(ValueError):
            m.delta(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(-0.2))
        result = m.delta(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0))
        expect = torch.full_like(result, 1.0 if call else 0.0)
        assert_close(result, expect)
        result = m.delta(torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.1))
        expect = torch.full_like(result, 1.0 if call else 0.0)
        assert_close(result, expect)
        result = m.delta(torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0))
        expect = torch.full_like(result, 1.0 if call else 0.0)
        assert_close(result, expect)
        result = m.delta(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.1))
        expect = torch.full_like(result, 0.5 if call else -0.5)
        assert_close(result, expect)
        result = m.delta(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        expect = torch.full_like(result, 0.5 if call else -0.5)
        assert_close(result, expect)

    def test_gamma_1(self):
        m = BSEuropeanOption()
        result = m.gamma(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.2))
        expect = torch.full_like(result, 1.9847627374)
        assert_close(result, expect)

    def test_gamma_2(self):
        m = BSEuropeanOption(call=False)
        result = m.gamma(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.2))
        expect = torch.full_like(result, 1.9847627374)
        assert_close(result, expect)

    @pytest.mark.parametrize("call", [True, False])
    def test_gamma_3(self, call: bool):
        m = BSEuropeanOption(call=call)
        with pytest.raises(ValueError):
            m.gamma(torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(0.2))
        with pytest.raises(ValueError):
            m.gamma(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(-0.2))
        result = m.gamma(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.gamma(torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.1))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.gamma(torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.gamma(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.1))
        expect = torch.full_like(result, float("inf"))
        assert_close(result, expect)
        result = m.gamma(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        expect = torch.full_like(result, float("inf"))
        assert_close(result, expect)

    def test_price_1(self):
        m = BSEuropeanOption()
        result = m.price(0.0, 1.0, 0.2)
        expect = torch.full_like(result, 0.0796557924)
        assert_close(result, expect)

    def test_price_2(self):
        m = BSEuropeanOption(call=False)
        result = m.price(0.0, 1.0, 0.2)
        expect = torch.full_like(result, 0.0796557924)
        assert_close(result, expect)

    @pytest.mark.parametrize("call", [True, False])
    def test_price_3(self, call: bool):
        m = BSEuropeanOption(call=call)
        with pytest.raises(ValueError):
            m.price(torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(0.2))
        with pytest.raises(ValueError):
            m.price(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(-0.2))
        result = m.price(
            torch.tensor(1.0 if call else -1.0), torch.tensor(1.0), torch.tensor(0)
        )
        expect = torch.full_like(result, 1.718282 if call else 0.632121)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(1.0 if call else -1.0), torch.tensor(0.0), torch.tensor(0.1)
        )
        expect = torch.full_like(result, 1.718282 if call else 0.632121)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(1.0 if call else -1.0), torch.tensor(0.0), torch.tensor(0.0)
        )
        expect = torch.full_like(result, 1.718282 if call else 0.632121)
        assert_close(result, expect)
        result = m.price(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.1))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.price(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)

    def test_implied_volatility(self):
        input = torch.tensor([[0.0, 0.1, 0.01], [0.0, 0.1, 0.02], [0.0, 0.1, 0.03]])
        m = BSEuropeanOption()
        iv = m.implied_volatility(input[:, 0], input[:, 1], input[:, 2])

        result = BSEuropeanOption().price(input[:, 0], input[:, 1], iv)
        expect = input[:, 2]
        assert_close(result, expect, check_stride=False)

    def test_vega(self):
        input = torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.2, 0.2], [0.0, 0.3, 0.2]])
        m = BSEuropeanOption()
        result = m.vega(
            log_moneyness=input[..., 0],
            time_to_maturity=input[..., 1],
            volatility=input[..., 2],
        )
        expect = torch.tensor([0.1261, 0.1782, 0.2182])
        assert_close(result, expect, atol=1e-3, rtol=0)

    @pytest.mark.parametrize("call", [True, False])
    def test_vega_2(self, call: bool):
        m = BSEuropeanOption(call=call)
        with pytest.raises(ValueError):
            m.vega(torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(0.2))
        with pytest.raises(ValueError):
            m.vega(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(-0.2))
        result = m.vega(torch.tensor(0.1), torch.tensor(1.0), torch.tensor(0))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.vega(torch.tensor(0.1), torch.tensor(0.0), torch.tensor(0.1))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.vega(torch.tensor(0.1), torch.tensor(0.0), torch.tensor(0.0))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.vega(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.1))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.vega(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)

    def test_vega_and_gamma(self):
        m = BSEuropeanOption()
        # vega = spot^2 * sigma * (T - t) * gamma
        # See Chapter 5 Appendix A, Bergomi "Stochastic volatility modeling"
        spot = torch.tensor([0.9, 1.0, 1.1, 1.1, 1.1, 1.1])
        t = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.1])
        v = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.2, 0.0])
        vega = m.vega(spot.log(), t, v)
        gamma = m.gamma(spot.log(), t, v)
        assert_close(vega, spot.square() * v * t * gamma, atol=1e-3, rtol=0)

    def test_theta(self):
        input = torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.2, 0.2], [0.0, 0.3, 0.2]])
        m = BSEuropeanOption(strike=100)
        result = m.theta(
            log_moneyness=input[..., 0],
            time_to_maturity=input[..., 1],
            volatility=input[..., 2],
        )
        expect = torch.tensor([-12.6094, -8.9117, -7.2727])
        assert_close(result, expect, atol=1e-3, rtol=0)

    @pytest.mark.parametrize("call", [True, False])
    def test_theta_2(self, call: bool):
        m = BSEuropeanOption(call=call)
        with pytest.raises(ValueError):
            m.theta(torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(0.2))
        with pytest.raises(ValueError):
            m.theta(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(-0.2))
        result = m.theta(torch.tensor(0.1), torch.tensor(1.0), torch.tensor(0))
        expect = torch.full_like(result, -0.0)
        assert_close(result, expect)
        result = m.theta(torch.tensor(0.1), torch.tensor(0.0), torch.tensor(0.1))
        expect = torch.full_like(result, -0.0)
        assert_close(result, expect)
        result = m.theta(torch.tensor(0.1), torch.tensor(0.0), torch.tensor(0.0))
        expect = torch.full_like(result, -0.0)
        assert_close(result, expect)
        result = m.theta(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.1))
        expect = torch.full_like(result, -float("inf"))
        assert_close(result, expect)
        result = m.theta(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        expect = torch.full_like(result, -0.0)
        assert_close(result, expect)

    def test_example(self):
        from pfhedge.instruments import BrownianStock
        from pfhedge.instruments import EuropeanOption
        from pfhedge.nn import Hedger

        derivative = EuropeanOption(BrownianStock())
        model = BSEuropeanOption()
        hedger = Hedger(model, model.inputs())
        result = hedger.price(derivative)
        expect = torch.tensor(0.022)
        assert_close(result, expect, atol=1e-3, rtol=1e-3)

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSEuropeanOption()
        self.assert_shape_delta(m)
        self.assert_shape_gamma(m)
        self.assert_shape_vega(m)
        self.assert_shape_theta(m)
        self.assert_shape_price(m)
        self.assert_shape_forward(m)
