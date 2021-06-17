import pytest
import torch
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanBinaryOption
from pfhedge.nn import BSEuropeanBinaryOption

from ._base import _TestBSModule


class TestBSEuropeanBinaryOption(_TestBSModule):
    """
    pfhedge.nn.bs.BSEuropeanBinaryOption
    """

    def setup_class(self):
        torch.manual_seed(42)

    def test_repr(self):
        m = BSEuropeanBinaryOption()
        assert repr(m) == "BSEuropeanBinaryOption()"

        derivative = EuropeanBinaryOption(BrownianStock(), strike=1.1)
        m = BSEuropeanBinaryOption.from_derivative(derivative)
        assert repr(m) == "BSEuropeanBinaryOption(strike=1.1)"

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
        assert m.inputs() == ["log_moneyness", "expiry_time", "volatility"]
        _ = [get_feature(f) for f in m.inputs()]

    def test_forward(self):
        m = BSEuropeanBinaryOption()
        input = torch.tensor([0.0, 0.1, 0.2]).reshape(1, -1)
        result = m(input)
        expect = torch.full_like(result, 6.3047)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_delta(self):
        m = BSEuropeanBinaryOption()
        result = m.delta(0.0, 0.1, 0.2)
        expect = torch.tensor(6.3047)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_gamma(self):
        m = BSEuropeanBinaryOption()
        result = m.gamma(0.01, 1.0, 0.2)
        expect = torch.tensor(-1.4645787477493286)
        assert_close(result, expect)

        with pytest.raises(ValueError):
            # not yet supported
            m = BSEuropeanBinaryOption(call=False)
            result = m.gamma(0.0, 1.0, 0.2)

    def test_price(self):
        m = BSEuropeanBinaryOption()

        result = m.price(0.0, 0.1, 0.2)
        expect = torch.tensor(0.4874)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        result = m.price(0.0001, 0.1, 0.2)
        expect = torch.tensor(0.4880)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

    def test_implied_volatility(self):
        # log_moneyness, expiry_time, price
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

        deriv = EuropeanBinaryOption(BrownianStock())
        model = BSEuropeanBinaryOption(deriv)
        hedger = Hedger(model, model.inputs())
        result = hedger.price(deriv)
        expect = torch.tensor(0.51)
        assert_close(result, expect, atol=1e-2, rtol=1e-2)

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSEuropeanBinaryOption()
        self.assert_shape_delta(m)
        # self.assert_shape_gamma(m)
        self.assert_shape_price(m)
        self.assert_shape_forward(m)
