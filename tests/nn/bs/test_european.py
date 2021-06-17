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

    def test_repr(self):
        m = BSEuropeanOption()
        assert repr(m) == "BSEuropeanOption()"

        liability = EuropeanOption(BrownianStock(), strike=1.1, call=False)
        m = BSEuropeanOption(liability)
        assert repr(m) == "BSEuropeanOption(call=False, strike=1.1)"

    def test_features(self):
        m = BSEuropeanOption()
        assert m.inputs() == ["log_moneyness", "expiry_time", "volatility"]
        _ = [get_feature(f) for f in m.inputs()]

    def test_forward(self):
        m = BSEuropeanOption()
        input = torch.tensor([0.0, 1.0, 0.2]).reshape(1, -1)
        result = m(input)
        expect = torch.full_like(result, 0.5398278962)
        assert_close(result, expect)

        m = BSEuropeanOption(call=False)
        input = torch.tensor([0.0, 1.0, 0.2]).reshape(1, -1)
        result = m(input)
        expect = torch.full_like(result, -0.4601721)
        assert_close(result, expect)

        liability = EuropeanOption(BrownianStock(), call=False)
        m = BSEuropeanOption(liability)
        input = torch.tensor([0.0, 1.0, 0.2]).reshape(1, -1)
        result = m(input)
        expect = torch.full_like(result, -0.4601721)
        assert_close(result, expect)

    def test_delta(self):
        m = BSEuropeanOption()
        result = m.delta(0.0, 1.0, 0.2)
        expect = torch.full_like(result, 0.5398278962)
        assert_close(result, expect)

        m = BSEuropeanOption(call=False)
        result = m.delta(0.0, 1.0, 0.2)
        expect = torch.full_like(result, -0.4601721)
        assert_close(result, expect)

    def test_gamma(self):
        m = BSEuropeanOption()
        result = m.gamma(0.0, 1.0, 0.2)
        expect = torch.full_like(result, 1.9847627374)
        assert_close(result, expect)

        m = BSEuropeanOption(call=False)
        with pytest.raises(ValueError):
            # not yet supported
            result = m.gamma(0.0, 1.0, 0.2)

    def test_price(self):
        m = BSEuropeanOption()
        result = m.price(0.0, 1.0, 0.2)
        expect = torch.full_like(result, 0.0796557924)
        assert_close(result, expect)

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

        liability = EuropeanOption(BrownianStock())
        model = BSEuropeanOption()
        hedger = Hedger(model, model.inputs())
        result = hedger.price(liability)
        expect = torch.tensor(0.022)
        assert_close(result, expect, atol=1e-3, rtol=1e-3)

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSEuropeanOption()
        self.assert_shape_delta(m)
        self.assert_shape_gamma(m)
        self.assert_shape_price(m)
        self.assert_shape_forward(m)
