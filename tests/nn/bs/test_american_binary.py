import pytest
import torch
from torch import Tensor
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import AmericanBinaryOption
from pfhedge.instruments import BrownianStock
from pfhedge.nn import BSAmericanBinaryOption

from ._base import _TestBSModule
from ._utils import compute_delta
from ._utils import compute_gamma
from ._utils import compute_price


class TestBSAmericanBinaryOption(_TestBSModule):
    def test_repr(self):
        m = BSAmericanBinaryOption()
        assert repr(m) == "BSAmericanBinaryOption(strike=1.)"

        derivative = AmericanBinaryOption(BrownianStock(), strike=1.1)
        m = BSAmericanBinaryOption.from_derivative(derivative)
        assert repr(m) == "BSAmericanBinaryOption(strike=1.1000)"

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

    def test_check_delta(self):
        m = BSAmericanBinaryOption()

        # delta = 0 for max --> +0
        result = compute_delta(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for max > 0
        result = compute_delta(m, torch.tensor([[0.0, 0.01, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for time --> +0
        result = compute_delta(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for volatility --> +0
        result = compute_delta(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    def test_check_gamma(self):
        m = BSAmericanBinaryOption()

        # gamma = 0 for max --> +0
        result = compute_gamma(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for max > 0
        result = compute_gamma(m, torch.tensor([[0.0, 0.01, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for time --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for volatility --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    def test_check_price(self):
        m = BSAmericanBinaryOption()

        # price = 0 for max --> +0
        result = compute_price(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = 1 for max > 0
        result = compute_price(m, torch.tensor([[0.0, 0.01, 0.1, 0.2]]))
        expect = torch.tensor([1.0])
        assert_close(result, expect)

        # price = 0 for time --> +0
        result = compute_price(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # price = 0 for volatility --> +0
        result = compute_price(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    def test_check_price_monte_carlo(self):
        torch.manual_seed(42)

        # Monte Carlo evaluation of a lookback option needs small dt
        k = 1.01
        d = AmericanBinaryOption(BrownianStock(dt=1e-5), strike=k)
        m = BSAmericanBinaryOption.from_derivative(d)
        d.simulate(n_paths=int(1e4), init_state=(1.0,))

        s = torch.tensor([1.0 / k]).log()
        input = torch.tensor([[s, s, d.maturity, d.ul().sigma]])
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

    def test_features(self):
        m = BSAmericanBinaryOption()
        assert m.inputs() == [
            "log_moneyness",
            "max_log_moneyness",
            "time_to_maturity",
            "volatility",
        ]
        _ = [get_feature(f) for f in m.inputs()]

    def test_implied_volatility(self):
        # log_moneyness, max_log_moneyness, time_to_maturity, price
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

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSAmericanBinaryOption()
        self.assert_shape_delta(m)
        self.assert_shape_gamma(m)
        self.assert_shape_price(m)
        self.assert_shape_forward(m)
