import pytest
import torch
from torch import Tensor
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import LookbackOption
from pfhedge.nn import BSLookbackOption

from ._base import _TestBSModule
from ._utils import compute_delta
from ._utils import compute_gamma
from ._utils import compute_price


class TestBSLookbackOption(_TestBSModule):
    def test_repr(self):
        m = BSLookbackOption()
        assert repr(m) == "BSLookbackOption(strike=1.)"

        derivative = LookbackOption(BrownianStock(), strike=1.1)
        m = BSLookbackOption.from_derivative(derivative)
        assert repr(m) == "BSLookbackOption(strike=1.1000)"

    def test_features(self):
        m = BSLookbackOption()
        assert m.inputs() == [
            "log_moneyness",
            "max_log_moneyness",
            "time_to_maturity",
            "volatility",
        ]
        _ = [get_feature(f) for f in m.inputs()]

    def test_check_delta(self):
        m = BSLookbackOption()

        # delta = 0 for max --> +0
        result = compute_delta(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for max / spot --> +inf
        result = compute_delta(m, torch.tensor([[0.0, 10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for time --> +0
        result = compute_delta(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_delta(m, torch.tensor([[0.0, 0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for volatility --> +0
        result = compute_delta(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_delta(m, torch.tensor([[0.0, 0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    def test_check_gamma(self):
        m = BSLookbackOption()

        # gamma = 0 for max --> +0
        result = compute_gamma(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for max / spot --> +inf
        result = compute_gamma(m, torch.tensor([[0.0, 10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for time --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_gamma(m, torch.tensor([[0.0, 0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for spot / k < 1 and volatility --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_gamma(m, torch.tensor([[0.0, 0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    def test_check_price(self):
        m = BSLookbackOption()

        # price = 0 for max --> +0
        result = compute_price(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = max - strike for max --> +inf
        result = compute_price(m, torch.tensor([[0.0, 10.0, 0.1, 0.2]]))
        expect = torch.tensor([10.0]).exp() - m.strike
        assert_close(result, expect)

        # price = 0 for max < strike and time --> +0
        result = compute_price(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = max - strike for max > strike and time --> +0
        result = compute_price(m, torch.tensor([[0.0, 0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.01]).exp() - m.strike
        assert_close(result, expect)

        # price = 0 for spot < strike and volatility --> +0
        result = compute_price(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = max - strike for spot > strike and volatility --> +0
        result = compute_price(m, torch.tensor([[0.0, 0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.01]).exp() - m.strike
        assert_close(result, expect)

        # prices are almost equal for max = strike +- epsilon
        result0 = compute_price(m, torch.tensor([[-1e-5, -1e-5, 0.1, 0.2]]))
        result1 = compute_price(m, torch.tensor([[1e-5, 1e-5, 0.1, 0.2]]))
        assert_close(result0, result1, atol=1e-4, rtol=0)

    def test_check_price_monte_carlo(self):
        torch.manual_seed(42)

        # Monte Carlo evaluation of a lookback option needs small dt
        k = 1.01
        d = LookbackOption(BrownianStock(dt=1e-5), strike=k)
        m = BSLookbackOption.from_derivative(d)
        d.simulate(n_paths=int(1e4), init_state=(1.0,))

        s = torch.tensor([1.0 / k]).log()
        input = torch.tensor([[s, s, d.maturity, d.ul().sigma]])
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

    def test_forward(self):
        m = BSLookbackOption()
        s = torch.tensor(1.00 / 1.03).log()
        input = torch.tensor([[s, s, 1.0, 0.2]])
        result = m(input)
        expect = torch.full_like(result, 1.037)
        assert_close(result, expect, atol=1e-2, rtol=1e-2)

    def test_price(self):
        m = BSLookbackOption(strike=1.03)
        s = torch.tensor(1.00 / 1.03).log()
        result = m.price(s, s, torch.tensor(1.0), torch.tensor(0.2))
        expect = torch.tensor(0.14)
        assert_close(result, expect, atol=1e-2, rtol=1e-2)

    def test_delta(self):
        m = BSLookbackOption()
        s = torch.tensor(1.00 / 1.03).log()
        result = m.delta(s, s, torch.tensor(1.0), torch.tensor(0.2))
        expect = torch.tensor(1.037)
        assert_close(result, expect, atol=1e-2, rtol=1e-2)

    def test_gamma(self):
        m = BSLookbackOption()
        s = torch.tensor(1.00 / 1.03).log()
        result = m.gamma(s, s, torch.tensor(1.0), torch.tensor(0.2))
        expect = torch.tensor(4.466)
        assert_close(result, expect, atol=0.8, rtol=0.8)  # ?

    def test_implied_volatility(self):
        input = torch.tensor(
            [[0.0, 0.0, 0.1, 0.01], [0.0, 0.0, 0.1, 0.02], [0.0, 0.0, 0.1, 0.03]]
        )
        m = BSLookbackOption()
        iv = m.implied_volatility(input[:, 0], input[:, 1], input[:, 2], input[:, 3])
        result = BSLookbackOption().price(input[:, 0], input[:, 1], input[:, 2], iv)
        expect = input[:, -1]
        assert_close(result, expect, atol=1e-4, rtol=1e-4, check_stride=False)

    def test_put_notimplemented(self):
        with pytest.raises(ValueError):
            # not yet supported
            BSLookbackOption(call=False)

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSLookbackOption()
        self.assert_shape_delta(m)
        self.assert_shape_gamma(m)
        self.assert_shape_price(m)
        self.assert_shape_forward(m)

    def test_example(self):
        torch.manual_seed(42)

        from pfhedge.instruments import BrownianStock
        from pfhedge.instruments import LookbackOption
        from pfhedge.nn import Hedger

        deriv = LookbackOption(BrownianStock(), strike=1.03)
        model = BSLookbackOption.from_derivative(deriv)
        hedger = Hedger(model, model.inputs())
        price = hedger.price(deriv)

        assert_close(price, torch.tensor(0.017), atol=1e-3, rtol=1e-3)
