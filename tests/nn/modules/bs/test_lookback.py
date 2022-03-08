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
from ._utils import compute_theta
from ._utils import compute_vega


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

    @pytest.mark.parametrize("call", [True])
    def test_delta_2(self, call: bool):
        m = BSLookbackOption(call=call)
        with pytest.raises(ValueError):
            m.delta(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-1.0),
                torch.tensor(0.2),
            )
        with pytest.raises(ValueError):
            m.delta(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(1.0),
                torch.tensor(-0.2),
            )

    @pytest.mark.parametrize("call", [True])
    def test_delta_3(self, call: bool):
        derivative = LookbackOption(BrownianStock(), call=call)
        m = BSLookbackOption.from_derivative(derivative)
        m2 = BSLookbackOption(call=call)
        with pytest.raises(AttributeError):
            m.delta()
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.delta()
        expect = m2.delta(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        # ToDo: [..., :-1] should be removed
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.delta(
            None,
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.delta(
            derivative.log_moneyness(),
            None,
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.delta(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            None,
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.delta(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            None,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        with pytest.raises(ValueError):
            m2.delta(
                None,
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.delta(
                derivative.log_moneyness(),
                None,
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.delta(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                None,
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.delta(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                None,
            )

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

    @pytest.mark.parametrize("call", [True])
    def test_gamma_2(self, call: bool):
        m = BSLookbackOption(call=call)
        with pytest.raises(ValueError):
            m.gamma(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-1.0),
                torch.tensor(0.2),
            )
        with pytest.raises(ValueError):
            m.gamma(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(1.0),
                torch.tensor(-0.2),
            )

    @pytest.mark.parametrize("call", [True])
    def test_gamma_3(self, call: bool):
        derivative = LookbackOption(BrownianStock(), call=call)
        m = BSLookbackOption.from_derivative(derivative)
        m2 = BSLookbackOption(call=call)
        with pytest.raises(AttributeError):
            m.gamma()
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.gamma()
        expect = m2.gamma(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        # ToDo: [..., :-1] should be removed
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.gamma(
            None,
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.gamma(
            derivative.log_moneyness(),
            None,
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.gamma(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            None,
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.gamma(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            None,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        with pytest.raises(ValueError):
            m2.gamma(
                None,
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.gamma(
                derivative.log_moneyness(),
                None,
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.gamma(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                None,
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.gamma(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                None,
            )

    def test_check_vega(self):
        m = BSLookbackOption()

        # gamma = 0 for max --> +0
        result = compute_vega(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for max / spot --> +inf
        result = compute_vega(m, torch.tensor([[0.0, 10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for time --> +0
        result = compute_vega(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_vega(m, torch.tensor([[0.0, 0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for spot / k < 1 and volatility --> +0
        result = compute_vega(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_vega(m, torch.tensor([[0.0, 0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    @pytest.mark.parametrize("call", [True])
    def test_vega_2(self, call: bool):
        m = BSLookbackOption(call=call)
        with pytest.raises(ValueError):
            m.vega(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-1.0),
                torch.tensor(0.2),
            )
        with pytest.raises(ValueError):
            m.vega(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(1.0),
                torch.tensor(-0.2),
            )

    @pytest.mark.parametrize("call", [True])
    def test_vega_3(self, call: bool):
        derivative = LookbackOption(BrownianStock(), call=call)
        m = BSLookbackOption.from_derivative(derivative)
        m2 = BSLookbackOption(call=call)
        with pytest.raises(AttributeError):
            m.vega()
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.vega()
        expect = m2.vega(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        # ToDo: [..., :-1] should be removed
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.vega(
            None,
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.vega(
            derivative.log_moneyness(),
            None,
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.vega(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            None,
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.vega(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            None,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        with pytest.raises(ValueError):
            m2.vega(
                None,
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.vega(
                derivative.log_moneyness(),
                None,
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.vega(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                None,
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.vega(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                None,
            )

    def test_check_theta(self):
        m = BSLookbackOption()

        # gamma = 0 for max --> +0
        result = compute_theta(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # gamma = 0 for max / spot --> +inf
        result = compute_theta(m, torch.tensor([[0.0, 10.0, 0.1, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for time --> +0
        result = compute_theta(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_theta(m, torch.tensor([[0.0, 0.01, 1e-10, 0.2]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        # delta = 0 for spot / k < 1 and volatility --> +0
        result = compute_theta(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

        result = compute_theta(m, torch.tensor([[0.0, 0.01, 0.1, 1e-10]]))
        expect = torch.tensor([0.0])
        assert_close(result, expect)

    @pytest.mark.parametrize("call", [True])
    def test_theta_2(self, call: bool):
        m = BSLookbackOption(call=call)
        with pytest.raises(ValueError):
            m.theta(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-1.0),
                torch.tensor(0.2),
            )
        with pytest.raises(ValueError):
            m.theta(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(1.0),
                torch.tensor(-0.2),
            )

    @pytest.mark.parametrize("call", [True])
    def test_theta_3(self, call: bool):
        derivative = LookbackOption(BrownianStock(), call=call)
        m = BSLookbackOption.from_derivative(derivative)
        m2 = BSLookbackOption(call=call)
        with pytest.raises(AttributeError):
            m.theta()
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.theta()
        expect = m2.theta(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        # ToDo: [..., :-1] should be removed
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.theta(
            None,
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.theta(
            derivative.log_moneyness(),
            None,
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.theta(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            None,
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.theta(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            None,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        with pytest.raises(ValueError):
            m2.theta(
                None,
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.theta(
                derivative.log_moneyness(),
                None,
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.theta(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                None,
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.theta(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                None,
            )

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

    @pytest.mark.parametrize("call", [True])
    def test_price_3(self, call: bool):
        m = BSLookbackOption(call=call)
        with pytest.raises(ValueError):
            m.price(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(-1.0),
                torch.tensor(0.2),
            )
        with pytest.raises(ValueError):
            m.price(
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(1.0),
                torch.tensor(-0.2),
            )

    @pytest.mark.parametrize("call", [True])
    def test_price_4(self, call: bool):
        derivative = LookbackOption(BrownianStock(), call=call)
        m = BSLookbackOption.from_derivative(derivative)
        m2 = BSLookbackOption(call=call)
        with pytest.raises(AttributeError):
            m.price()
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.price()
        expect = m2.price(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        # ToDo: [..., :-1] should be removed
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.price(
            None,
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.price(
            derivative.log_moneyness(),
            None,
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.price(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            None,
            derivative.underlier.volatility,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.price(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            None,
        )
        assert_close(result[..., :-1], expect[..., :-1])
        with pytest.raises(ValueError):
            m2.price(
                None,
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.price(
                derivative.log_moneyness(),
                None,
                derivative.time_to_maturity(),
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.price(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                None,
                derivative.underlier.volatility,
            )
        with pytest.raises(ValueError):
            m2.price(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                None,
            )

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

    @pytest.mark.parametrize("call", [True])
    def test_implied_volatility_2(self, call: bool):
        derivative = LookbackOption(BrownianStock(), call=call)
        m = BSLookbackOption.from_derivative(derivative)
        m2 = BSLookbackOption(call=call)
        with pytest.raises(AttributeError):
            m.implied_volatility()
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        with pytest.raises(ValueError):
            m.implied_volatility()
        result = m.implied_volatility(price=derivative.underlier.spot)
        expect = m2.implied_volatility(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.spot,
        )
        assert_close(result, expect)
        result = m.implied_volatility(
            None,
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.spot,
        )
        assert_close(result, expect)
        result = m.implied_volatility(
            derivative.log_moneyness(),
            None,
            derivative.time_to_maturity(),
            derivative.underlier.spot,
        )
        assert_close(result, expect)
        result = m.implied_volatility(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            None,
            derivative.underlier.spot,
        )
        assert_close(result, expect)
        with pytest.raises(ValueError):
            m2.implied_volatility(
                None,
                derivative.max_log_moneyness(),
                derivative.time_to_maturity(),
                derivative.underlier.spot,
            )
        with pytest.raises(ValueError):
            m2.implied_volatility(
                derivative.log_moneyness(),
                None,
                derivative.time_to_maturity(),
                derivative.underlier.spot,
            )
        with pytest.raises(ValueError):
            m2.implied_volatility(
                derivative.log_moneyness(),
                derivative.max_log_moneyness(),
                None,
                derivative.underlier.spot,
            )

    def test_vega_and_gamma(self):
        m = BSLookbackOption()
        # vega = spot^2 * sigma * (T - t) * gamma
        # See Chapter 5 Appendix A, Bergomi "Stochastic volatility modeling"
        spot = torch.tensor([0.90, 0.94, 0.98])
        t = torch.tensor([0.1, 0.2, 0.3])
        v = torch.tensor([0.1, 0.2, 0.3])
        vega = m.vega(spot.log(), spot.log(), t, v)
        gamma = m.gamma(spot.log(), spot.log(), t, v)
        assert_close(vega, spot.square() * v * t * gamma, atol=1e-3, rtol=0)

    @pytest.mark.parametrize("call", [True])
    def test_vega_and_gamma_2(self, call: bool):
        derivative = LookbackOption(BrownianStock(), call=call)
        m = BSLookbackOption.from_derivative(derivative)
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        vega = m.vega()
        gamma = m.gamma()
        # ToDo: [..., :-1] should be removed
        assert_close(
            vega[..., :-1],
            (
                derivative.underlier.spot.square()
                * derivative.underlier.volatility
                * derivative.time_to_maturity()
                * gamma
            )[..., :-1],
            atol=1e-3,
            rtol=0,
        )

    def test_put_notimplemented(self):
        with pytest.raises(ValueError):
            # not yet supported
            BSLookbackOption(call=False)

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSLookbackOption()
        self.assert_shape_delta(m)
        self.assert_shape_gamma(m)
        self.assert_shape_vega(m)
        self.assert_shape_theta(m)
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
