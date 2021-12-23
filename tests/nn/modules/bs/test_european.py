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

    def test_check_delta(self):
        # TODO(simaki): Check for put option

        m = BSEuropeanOption()

        # delta = 0 for spot --> +0
        result = compute_delta(m, torch.tensor([[-10.0, 1.0, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 1 for spot --> +inf
        result = compute_delta(m, torch.tensor([[10.0, 1.0, 0.2]]))
        expect = torch.tensor([1.0])
        assert_close(result, expect)

        # delta = 0 for spot / k < 1 and time --> +0
        result = compute_delta(m, torch.tensor([[-0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 1 for spot / k > 1 and time --> +0
        result = compute_delta(m, torch.tensor([[0.01, 1e-10, 0.2]]))
        expect = torch.tensor([1.0])
        assert_close(result, expect)

        # delta = 0 for spot / k < 1 and volatility --> +0
        result = compute_delta(m, torch.tensor([[-0.01, 1.0, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for spot / k > 1 and volatility --> +0
        result = compute_delta(m, torch.tensor([[0.01, 1.0, 1e-10]]))
        expect = torch.tensor([1.0])
        assert_close(result, expect)

    def test_check_gamma(self):
        # TODO(simaki): Check for put option

        m = BSEuropeanOption()

        # gamma = 0 for spot --> +0
        result = compute_gamma(m, torch.tensor([[-10.0, 1.0, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 1 for spot --> +inf
        result = compute_gamma(m, torch.tensor([[10.0, 1.0, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for spot / k < 1 and time --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for spot / k > 1 and time --> +0
        result = compute_gamma(m, torch.tensor([[0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for spot / k < 1 and volatility --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, 1.0, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for spot / k > 1 and volatility --> +0
        result = compute_gamma(m, torch.tensor([[0.01, 1.0, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    def test_check_price(self):
        # TODO(simaki): Check for put option

        m = BSEuropeanOption()

        # price = 0 for spot --> +0
        result = compute_price(m, torch.tensor([[-10.0, 1.0, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = spot - k for spot --> +inf
        result = compute_price(m, torch.tensor([[10.0, 1.0, 0.2]]))
        expect = torch.tensor([torch.tensor([10.0]).exp() - 1.0])
        assert_close(result, expect)

        # price = 0 for spot / k < 1 and time --> +0
        result = compute_price(m, torch.tensor([[-0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = spot - k for spot / k > 1 and time --> +0
        result = compute_price(m, torch.tensor([[0.01, 1e-10, 0.2]]))
        expect = torch.tensor([torch.tensor(0.01).exp() - 1.0])
        assert_close(result, expect)

        # price = 0 for spot / k < 1 and volatility --> +0
        result = compute_price(m, torch.tensor([[-0.01, 1.0, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = spot - k for spot / k > 1 and volatility --> +0
        result = compute_price(m, torch.tensor([[0.01, 1.0, 1e-10]]))
        expect = torch.tensor([torch.tensor(0.01).exp() - 1.0])
        assert_close(result, expect)

    def test_check_price_monte_carlo(self):
        torch.manual_seed(42)

        d = EuropeanOption(BrownianStock())
        m = BSEuropeanOption.from_derivative(d)
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

    def test_gamma_1(self):
        m = BSEuropeanOption()
        result = m.gamma(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.2))
        expect = torch.full_like(result, 1.9847627374)
        assert_close(result, expect)

    def test_gamma_2(self):
        m = BSEuropeanOption(call=False)
        with pytest.raises(ValueError):
            # not yet supported
            _ = m.gamma(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.2))

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
