import numpy as np
import pytest
import torch

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
        assert m.features() == ["log_moneyness", "expiry_time", "volatility"]
        _ = [get_feature(f) for f in m.features()]

    def test_forward(self):
        m = BSEuropeanOption()
        x = torch.tensor([0.0, 1.0, 0.2]).reshape(1, -1)
        result = m(x).item()
        assert np.isclose(result, 0.5398278962)

        m = BSEuropeanOption(call=False)
        x = torch.tensor([0.0, 1.0, 0.2]).reshape(1, -1)
        result = m(x).item()
        assert np.isclose(result, -0.4601721)

        liability = EuropeanOption(BrownianStock(), call=False)
        m = BSEuropeanOption(liability)
        x = torch.tensor([0.0, 1.0, 0.2]).reshape(1, -1)
        result = m(x).item()
        assert np.isclose(result, -0.4601721)

    def test_delta(self):
        m = BSEuropeanOption()
        result = m.delta(0.0, 1.0, 0.2).item()
        assert np.isclose(result, 0.5398278962)

        m = BSEuropeanOption(call=False)
        result = m.delta(0.0, 1.0, 0.2).item()
        assert np.isclose(result, -0.4601721)

    def test_gamma(self):
        m = BSEuropeanOption()
        result = m.gamma(0.0, 1.0, 0.2).item()
        assert np.isclose(result, 1.9847627374)

        m = BSEuropeanOption(call=False)
        with pytest.raises(ValueError):
            # not yet supported
            result = m.gamma(0.0, 1.0, 0.2).item()

    def test_price(self):
        m = BSEuropeanOption()
        result = m.price(0.0, 1.0, 0.2).item()
        assert np.isclose(result, 0.0796557924)

        m = BSEuropeanOption(call=False)
        result = m.price(0.0, 1.0, 0.2).item()
        assert np.isclose(result, 0.0796557924)

    def test_implied_volatility(self):
        x = torch.tensor([[0.0, 0.1, 0.01], [0.0, 0.1, 0.02], [0.0, 0.1, 0.03]])
        m = BSEuropeanOption()
        iv = m.implied_volatility(x[:, 0], x[:, 1], x[:, 2])

        result = BSEuropeanOption().price(x[:, 0], x[:, 1], iv)
        expect = x[:, 2]
        assert torch.allclose(result, expect)

    def test_example(self):
        from pfhedge import Hedger
        from pfhedge.instruments import BrownianStock
        from pfhedge.instruments import EuropeanOption

        liability = EuropeanOption(BrownianStock())
        model = BSEuropeanOption()
        hedger = Hedger(model, model.features())
        price = hedger.price(liability)
        assert torch.allclose(price, torch.tensor(0.0221), atol=1e-4)

    def test_shape(self):
        m = BSEuropeanOption()
        self.assert_shape_delta(m)
        self.assert_shape_gamma(m)
        self.assert_shape_price(m)
        self.assert_shape_forward(m)
