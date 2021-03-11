import numpy as np
import pytest
import torch

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

        liability = AmericanBinaryOption(BrownianStock(), strike=1.1)
        m = BSAmericanBinaryOption(liability)
        assert repr(m) == "BSAmericanBinaryOption(strike=1.1)"

        with pytest.raises(ValueError):
            # not yet supported
            liability = AmericanBinaryOption(BrownianStock(), strike=1.1, call=False)
            m = BSAmericanBinaryOption(liability)
            assert repr(m) == "BSAmericanBinaryOption(call=False, strike=1.1)"

    def test_error_put(self):
        with pytest.raises(ValueError):
            # not yet supported
            liability = AmericanBinaryOption(BrownianStock(), call=False)
            BSAmericanBinaryOption(liability)

    def test_features(self):
        m = BSAmericanBinaryOption()
        assert m.features() == [
            "log_moneyness",
            "max_log_moneyness",
            "expiry_time",
            "volatility",
        ]
        _ = [get_feature(f) for f in m.features()]

    def test_forward(self):
        m = BSAmericanBinaryOption()

        x = torch.tensor([-0.1, -0.1, 0.1, 0.2]).reshape(1, -1)
        result = m(x)
        expect = torch.tensor(0.7819)
        assert torch.allclose(result, expect, atol=1e-4)

        x = torch.tensor([0.1, 0.1, 0.1, 0.2]).reshape(1, -1)
        result = m(x)
        expect = torch.tensor(0.0)
        assert torch.allclose(result, expect, atol=1e-4)

        x = torch.tensor([-0.0001, -0.0001, 0.1, 0.2]).reshape(1, -1)
        result = m(x)
        expect = torch.tensor(1.1531)
        assert torch.allclose(result, expect, atol=1e-4)

    def test_delta(self):
        m = BSAmericanBinaryOption()

        result = m.delta(-0.1, -0.1, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(0.7819)
        assert torch.allclose(result, expect, atol=1e-4)

        result = m.delta(0.1, 0.1, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(0.0)
        assert torch.allclose(result, expect, atol=1e-4)

        result = m.delta(-0.0001, -0.0001, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(1.1531)
        assert torch.allclose(result, expect, atol=1e-4)

    def test_gamma(self):
        m = BSAmericanBinaryOption()
        result = m.gamma(-0.01, -0.01, 1.0, 0.2).item()
        assert np.isclose(result, 0.7618406414985657)

        with pytest.raises(ValueError):
            # not yet supported
            m = BSAmericanBinaryOption(call=False)
            result = m.gamma(-0.01, -0.01, 1.0, 0.2).item()

    def test_price(self):
        m = BSAmericanBinaryOption()
        result = m.price(-0.1, -0.1, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(0.5778)
        assert torch.allclose(result, expect, atol=1e-4)

        result = m.price(-0.0001, -0.0001, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(0.9995)
        assert torch.allclose(result, expect, atol=1e-4)

        result = m.price(0.1, 0.1, 0.1, 0.2)  # s, m, t, v
        expect = torch.tensor(1.0)
        assert torch.allclose(result, expect, atol=1e-4)

    def test_implied_volatility(self):
        # log_moneyness, max_log_moneyness, expiry_time, price
        x = torch.tensor(
            [
                [-0.01, -0.01, 0.1, 0.5],
                [-0.01, -0.01, 0.1, 0.6],
                [-0.01, -0.01, 0.1, 0.7],
            ]
        )
        m = BSAmericanBinaryOption()
        iv = m.implied_volatility(x[:, 0], x[:, 1], x[:, 2], x[:, 3])

        result = BSAmericanBinaryOption().price(x[:, 0], x[:, 1], x[:, 2], iv)
        expect = x[:, -1]
        print(result)
        print(expect)
        assert torch.allclose(result, expect, atol=1e-4)

    def test_example(self):
        from pfhedge import Hedger
        from pfhedge.instruments import AmericanBinaryOption
        from pfhedge.instruments import BrownianStock

        deriv = AmericanBinaryOption(BrownianStock(), strike=1.03)
        model = BSAmericanBinaryOption(deriv)
        hedger = Hedger(model, model.features())
        price = hedger.price(deriv)

        assert torch.allclose(price, torch.tensor(0.6219), atol=1e-4)

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSAmericanBinaryOption()
        self.assert_shape_delta(m)
        # self.assert_shape_gamma(m)
        self.assert_shape_price(m)
        self.assert_shape_forward(m)
