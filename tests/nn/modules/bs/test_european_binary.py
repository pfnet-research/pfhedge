import pytest
import torch
from torch import Tensor
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanBinaryOption
from pfhedge.nn import BSEuropeanBinaryOption

from ._base import _TestBSModule
from ._utils import compute_delta
from ._utils import compute_gamma
from ._utils import compute_price


class TestBSEuropeanBinaryOption(_TestBSModule):
    def setup_class(self):
        torch.manual_seed(42)

    def test_repr(self):
        m = BSEuropeanBinaryOption()
        assert repr(m) == "BSEuropeanBinaryOption(strike=1.)"

        derivative = EuropeanBinaryOption(BrownianStock(), strike=1.1)
        m = BSEuropeanBinaryOption.from_derivative(derivative)
        assert repr(m) == "BSEuropeanBinaryOption(strike=1.1000)"

        with pytest.raises(ValueError):
            # not yet supported
            derivative = EuropeanBinaryOption(BrownianStock(), strike=1.1, call=False)
            m = BSEuropeanBinaryOption.from_derivative(derivative)
            assert repr(m) == "BSEuropeanBinaryOption(call=False, strike=1.1)"

    def test_error_put(self):
        with pytest.raises(ValueError):
            # not yet supported
            derivative = EuropeanBinaryOption(BrownianStock(), call=False)
            BSEuropeanBinaryOption.from_derivative(derivative)

    def test_features(self):
        m = BSEuropeanBinaryOption()
        assert m.inputs() == ["log_moneyness", "time_to_maturity", "volatility"]
        _ = [get_feature(f) for f in m.inputs()]

    def test_check_delta(self):
        m = BSEuropeanBinaryOption()

        # delta = 0 for spot --> +0
        result = compute_delta(m, torch.tensor([[-10.0, 1.0, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for spot --> +inf
        result = compute_delta(m, torch.tensor([[10.0, 1.0, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for time --> +0
        result = compute_delta(m, torch.tensor([[-0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_delta(m, torch.tensor([[0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for volatility --> +0
        result = compute_delta(m, torch.tensor([[-0.01, 1.0, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_delta(m, torch.tensor([[0.01, 1.0, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    def test_check_gamma(self):
        m = BSEuropeanBinaryOption()

        # gamma = 0 for spot --> +0
        result = compute_gamma(m, torch.tensor([[-10.0, 1.0, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for spot --> +inf
        result = compute_gamma(m, torch.tensor([[10.0, 1.0, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for time --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_gamma(m, torch.tensor([[0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for volatility --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, 1.0, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_gamma(m, torch.tensor([[0.01, 1.0, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    def test_check_price(self):
        m = BSEuropeanBinaryOption()

        # price = 0 for spot --> +0
        result = compute_price(m, torch.tensor([[-10.0, 1.0, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = 1 for spot --> +inf
        result = compute_price(m, torch.tensor([[10.0, 1.0, 0.2]]))
        expect = torch.tensor([1.0])
        assert_close(result, expect)

        # price = 0 for spot < strike and time --> +0
        result = compute_price(m, torch.tensor([[-0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = 1 for spot > strike and time --> +0
        result = compute_price(m, torch.tensor([[0.01, 1e-10, 0.2]]))
        expect = torch.tensor([1.0])
        assert_close(result, expect)

        # price = 0 for spot < strike and volatility --> +0
        result = compute_price(m, torch.tensor([[-0.01, 1.0, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = 0 for spot > strike and volatility --> +0
        result = compute_price(m, torch.tensor([[0.01, 1.0, 1e-10]]))
        expect = torch.tensor([1.0])
        assert_close(result, expect)

    def test_check_price_monte_carlo(self):
        torch.manual_seed(42)

        d = EuropeanBinaryOption(BrownianStock())
        m = BSEuropeanBinaryOption.from_derivative(d)
        d.simulate(n_paths=int(1e6))

        input = torch.tensor([[0.0, d.maturity, d.ul().sigma]])
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

    def test_forward(self):
        m = BSEuropeanBinaryOption()
        input = torch.tensor([[0.0, 0.1, 0.2]])
        result = m(input)
        expect = torch.full_like(result, 6.3047)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_delta(self):
        m = BSEuropeanBinaryOption()
        result = m.delta(torch.tensor(0.0), torch.tensor(0.1), torch.tensor(0.2))
        expect = torch.tensor(6.3047)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        with pytest.raises(ValueError):
            # not yet supported
            m = BSEuropeanBinaryOption(call=False)
            result = m.gamma(torch.tensor(0.0), torch.tensor(0.1), torch.tensor(0.2))
            # expect = torch.tensor(-6.30468)
            # assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_gamma(self):
        m = BSEuropeanBinaryOption()
        result = m.gamma(torch.tensor(0.01), torch.tensor(1.0), torch.tensor(0.2))
        expect = torch.tensor(-1.4645787477493286)
        assert_close(result, expect)

        with pytest.raises(ValueError):
            # not yet supported
            m = BSEuropeanBinaryOption(call=False)
            result = m.gamma(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.2))
            # expect = torch.tensor(-6.30468)
            # assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_vega(self):
        m = BSEuropeanBinaryOption()
        result = m.vega(torch.tensor(0.0), torch.tensor(0.1), torch.tensor(0.2))
        expect = torch.tensor(-0.06305)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        with pytest.raises(ValueError):
            # not yet supported
            m = BSEuropeanBinaryOption(call=False)
            result = m.vega(torch.tensor(0.0), torch.tensor(0.1), torch.tensor(0.2))
            # expect = torch.tensor(0.06305)
            # assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_vega_and_gamma(self):
        m = BSEuropeanBinaryOption()
        # vega = spot^2 * sigma * (T - t) * gamma
        # See Chapter 5 Appendix A, Bergomi "Stochastic volatility modeling"
        spot = torch.tensor([0.9, 1.0, 1.1])
        t = torch.tensor([0.1, 0.2, 0.3])
        v = torch.tensor([0.1, 0.2, 0.3])
        vega = m.vega(spot.log(), t, v)
        gamma = m.gamma(spot.log(), t, v)
        assert_close(vega, spot.square() * v * t * gamma, atol=1e-3, rtol=0)

    def test_theta(self):
        m = BSEuropeanBinaryOption()
        result = m.theta(torch.tensor(0.0), torch.tensor(0.1), torch.tensor(0.2))
        expect = torch.tensor(0.0630)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        with pytest.raises(ValueError):
            # not yet supported
            m = BSEuropeanBinaryOption(call=False)
            result = m.theta(torch.tensor(0.0), torch.tensor(0.1), torch.tensor(0.2))
            # expect = torch.tensor(-0.0630)
            # assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_price(self):
        m = BSEuropeanBinaryOption()

        result = m.price(0.0, 0.1, 0.2)
        expect = torch.tensor(0.4874)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        result = m.price(0.0001, 0.1, 0.2)
        expect = torch.tensor(0.4880)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_implied_volatility(self):
        # log_moneyness, time_to_maturity, price
        input = torch.tensor(
            [[-0.01, 0.1, 0.40], [-0.01, 0.1, 0.41], [-0.01, 0.1, 0.42]]
        )
        m = BSEuropeanBinaryOption()
        iv = m.implied_volatility(input[:, 0], input[:, 1], input[:, 2])

        result = BSEuropeanBinaryOption().price(input[:, 0], input[:, 1], iv)
        expect = input[:, -1]
        assert_close(result, expect, check_stride=False)

    def test_example(self):
        from pfhedge.instruments import BrownianStock
        from pfhedge.instruments import EuropeanBinaryOption
        from pfhedge.nn import Hedger

        derivative = EuropeanBinaryOption(BrownianStock())
        model = BSEuropeanBinaryOption.from_derivative(derivative)
        hedger = Hedger(model, model.inputs())
        result = hedger.price(derivative)
        expect = torch.tensor(0.4922)
        x = hedger.compute_hedge(derivative)
        assert not x.isnan().any()
        assert_close(result, expect, atol=1e-2, rtol=1e-2)

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSEuropeanBinaryOption()
        self.assert_shape_delta(m)
        self.assert_shape_gamma(m)
        self.assert_shape_vega(m)
        self.assert_shape_theta(m)
        self.assert_shape_price(m)
        self.assert_shape_forward(m)
