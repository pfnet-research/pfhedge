import pytest
import torch
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import AmericanBinaryOption
from pfhedge.instruments import BrownianStock
from pfhedge.nn import BSAmericanBinaryOption

from ._base import _TestBSModule


class TestBSAmericanBinaryOption(_TestBSModule):
    """
    pfhedge.nn.bs.BSAmericanBinaryOption
    """

    def setup_class(self):
        torch.manual_seed(42)

    def test_repr(self):
        m = BSAmericanBinaryOption()
        assert repr(m) == "BSAmericanBinaryOption()"

        derivative = AmericanBinaryOption(BrownianStock(), strike=1.1)
        m = BSAmericanBinaryOption.from_derivative(derivative)
        assert repr(m) == "BSAmericanBinaryOption(strike=1.1)"

        with pytest.raises(ValueError):
            # not yet supported
            derivative = AmericanBinaryOption(BrownianStock(), strike=1.1, call=False)
            m = BSAmericanBinaryOption.from_derivative(derivative)
            assert repr(m) == "BSAmericanBinaryOption(call=False, strike=1.1)"

    def test_error_put(self):
        with pytest.raises(ValueError):
            # not yet supported
            derivative = AmericanBinaryOption(BrownianStock(), call=False)
            BSAmericanBinaryOption.from_derivative(derivative)

    def test_features(self):
        m = BSAmericanBinaryOption()
        assert m.inputs() == [
            "log_moneyness",
            "max_log_moneyness",
            "expiry_time",
            "volatility",
        ]
        _ = [get_feature(f) for f in m.inputs()]

    def test_forward(self):
        m = BSAmericanBinaryOption()

        input = torch.tensor([-0.1, -0.1, 0.1, 0.2]).reshape(1, -1)
        result = m(input)
        expect = torch.full_like(result, 0.7819)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        input = torch.tensor([0.1, 0.1, 0.1, 0.2]).reshape(1, -1)
        result = m(input)
        expect = torch.zeros_like(result)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        input = torch.tensor([-0.0001, -0.0001, 0.1, 0.2]).reshape(1, -1)
        result = m(input)
        expect = torch.full_like(result, 1.1531)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_delta(self):
        m = BSAmericanBinaryOption()

        result = m.delta(-0.1, -0.1, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(0.7819)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        result = m.delta(0.1, 0.1, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(0.0)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        result = m.delta(-0.0001, -0.0001, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(1.1531)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_gamma(self):
        m = BSAmericanBinaryOption()
        result = m.gamma(-0.01, -0.01, 1.0, 0.2)
        expect = torch.tensor(0.7618406414985657)
        assert_close(result, expect)

        with pytest.raises(ValueError):
            # not yet supported
            m = BSAmericanBinaryOption(call=False)
            result = m.gamma(-0.01, -0.01, 1.0, 0.2)

    def test_price(self):
        m = BSAmericanBinaryOption()
        result = m.price(-0.1, -0.1, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(0.5778)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        result = m.price(-0.0001, -0.0001, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(0.9995)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        result = m.price(0.1, 0.1, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(1.0)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_implied_volatility(self):
        # log_moneyness, max_log_moneyness, expiry_time, price
        input = torch.tensor(
            [
                [-0.01, -0.01, 0.1, 0.5],
                [-0.01, -0.01, 0.1, 0.6],
                [-0.01, -0.01, 0.1, 0.7],
            ]
        )
        m = BSAmericanBinaryOption()
        iv = m.implied_volatility(input[:, 0], input[:, 1], input[:, 2], input[:, 3])

        result = BSAmericanBinaryOption().price(
            input[:, 0], input[:, 1], input[:, 2], iv
        )
        expect = input[:, -1]
        assert_close(result, expect, atol=1e-4, rtol=1e-4, check_stride=False)

    def test_example(self):
        from pfhedge.instruments import AmericanBinaryOption
        from pfhedge.instruments import BrownianStock
        from pfhedge.nn import Hedger

        deriv = AmericanBinaryOption(BrownianStock(), strike=1.03)
        model = BSAmericanBinaryOption(deriv)
        hedger = Hedger(model, model.inputs())
        price = hedger.price(deriv)

        assert_close(price, torch.tensor(0.62), atol=1e-2, rtol=1e-4)

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSAmericanBinaryOption()
        self.assert_shape_delta(m)
        # self.assert_shape_gamma(m)
        self.assert_shape_price(m)
        self.assert_shape_forward(m)
