import pytest
import torch
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BSEuropeanOption

from ._base import _TestBSModule


class TestBSEuropeanOption(_TestBSModule):
    """
    pfhedge.nn.bs.BSEuropeanOption
    """

    def setup_class(self):
        torch.manual_seed(42)

    def test_repr_1(self):
        m = BSEuropeanOption()
        assert repr(m) == "BSEuropeanOption(strike=1.)"

    def test_repr_2(self):
        derivative = EuropeanOption(BrownianStock(), strike=1.1, call=False)
        m = BSEuropeanOption.from_derivative(derivative)
        assert repr(m) == "BSEuropeanOption(call=False, strike=1.1000)"

    def test_features(self):
        m = BSEuropeanOption()
        assert m.inputs() == ["log_moneyness", "time_to_maturity", "volatility"]
        _ = [get_feature(f) for f in m.inputs()]

    def test_forward_1(self):
        m = BSEuropeanOption()
        input = torch.tensor([[0.0, 1.0, 0.2]])
        result = m(input)
        expect = torch.full_like(result, 0.5398278962)
        assert_close(result, expect)

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
            result = m.gamma(0.0, 1.0, 0.2)

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
        self.assert_shape_price(m)
        self.assert_shape_forward(m)
