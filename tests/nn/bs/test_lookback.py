import pytest
import torch
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import LookbackOption
from pfhedge.nn import BSLookbackOption

from ._base import _TestBSModule


class TestBSLookbackOption(_TestBSModule):
    """
    pfhedge.nn.BSLookbackOption
    """

    def setup_class(self):
        torch.manual_seed(42)

    def test_repr(self):
        m = BSLookbackOption()
        assert repr(m) == "BSLookbackOption()"

        derivative = LookbackOption(BrownianStock(), strike=1.1)
        m = BSLookbackOption.from_derivative(derivative)
        assert repr(m) == "BSLookbackOption(strike=1.1)"

    def test_forward(self):
        m = BSLookbackOption()
        s = torch.tensor(1.00 / 1.03).log()
        input = torch.tensor([[s, s, 1.0, 0.2]])
        result = m(input)
        expect = torch.full_like(result, 1.037)
        assert_close(result, expect, atol=1e-2, rtol=1e-2)

    def test_features(self):
        m = BSLookbackOption()
        assert m.inputs() == [
            "log_moneyness",
            "max_log_moneyness",
            "time_to_maturity",
            "volatility",
        ]
        _ = [get_feature(f) for f in m.inputs()]

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
        print(result)
        print(expect)
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
        from pfhedge.instruments import BrownianStock
        from pfhedge.instruments import LookbackOption
        from pfhedge.nn import Hedger

        deriv = LookbackOption(BrownianStock(), strike=1.03)
        model = BSLookbackOption(deriv)
        hedger = Hedger(model, model.inputs())
        price = hedger.price(deriv)

        assert_close(price, torch.tensor(0.017), atol=1e-3, rtol=1e-3)
