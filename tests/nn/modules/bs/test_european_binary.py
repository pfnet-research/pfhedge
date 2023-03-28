import pytest
import torch
from torch.testing import assert_close

import pfhedge.autogreek as autogreek
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

        derivative = EuropeanBinaryOption(BrownianStock(), strike=1.1, call=False)
        m = BSEuropeanBinaryOption.from_derivative(derivative)
        assert repr(m) == "BSEuropeanBinaryOption(call=False, strike=1.1000)"

    def test_features(self):
        m = BSEuropeanBinaryOption()
        assert m.inputs() == ["log_moneyness", "time_to_maturity", "volatility"]
        _ = [get_feature(f) for f in m.inputs()]

    def test_delta_limit(self, device: str = "cpu"):
        c = BSEuropeanBinaryOption().to(device)
        p = BSEuropeanBinaryOption(call=False).to(device)

        # delta = 0 for spot --> +0
        result = compute_delta(c, torch.tensor([[-10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_delta(p, torch.tensor([[-10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # delta = 0 for spot --> +inf
        result = compute_delta(c, torch.tensor([[10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_delta(p, torch.tensor([[10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # delta = 0 for time --> +0
        result = compute_delta(c, torch.tensor([[-0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_delta(p, torch.tensor([[-0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        result = compute_delta(c, torch.tensor([[0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_delta(p, torch.tensor([[0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # delta = 0 for volatility --> +0
        result = compute_delta(c, torch.tensor([[-0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_delta(p, torch.tensor([[-0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        result = compute_delta(c, torch.tensor([[0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_delta(p, torch.tensor([[0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

    @pytest.mark.gpu
    def test_delta_limit_gpu(self):
        self.test_delta_limit(device="cuda")

    def test_gamma_limit(self, device: str = "cpu"):
        c = BSEuropeanBinaryOption().to(device)
        p = BSEuropeanBinaryOption(call=False).to(device)

        # gamma = 0 for spot --> +0
        result = compute_gamma(c, torch.tensor([[-10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[-10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # gamma = 0 for spot --> +inf
        result = compute_gamma(c, torch.tensor([[10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # gamma = 0 for time --> +0
        result = compute_gamma(c, torch.tensor([[-0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[-0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        result = compute_gamma(c, torch.tensor([[0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # gamma = 0 for volatility --> +0
        result = compute_gamma(c, torch.tensor([[-0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[-0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        result = compute_gamma(c, torch.tensor([[0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

    @pytest.mark.gpu
    def test_gamma_limit_gpu(self):
        self.test_gamma_limit(device="cuda")

    def test_price_limit(self, device: str = "cpu"):
        c = BSEuropeanBinaryOption().to(device)
        p = BSEuropeanBinaryOption(call=False).to(device)

        # price = 0 (call), 1 (put) for spot --> +0
        result = compute_price(c, torch.tensor([[-10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_price(p, torch.tensor([[-10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([1.0]).to(device))

        # price = 1 (call), 1 (put) for spot --> +inf
        result = compute_price(c, torch.tensor([[10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([1.0]).to(device))
        result = compute_price(p, torch.tensor([[10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # price = 0 (call), 1 (put) for spot < strike and time --> +0
        result = compute_price(c, torch.tensor([[-0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_price(p, torch.tensor([[-0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([1.0]).to(device))

        # price = 1 (call), 0 (put) for spot > strike and time --> +0
        result = compute_price(c, torch.tensor([[0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([1.0]).to(device))
        result = compute_price(p, torch.tensor([[0.01, 1e-10, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # price = 0 (call), 1 (put) for spot < strike and volatility --> +0
        result = compute_price(c, torch.tensor([[-0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_price(p, torch.tensor([[-0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([1.0]).to(device))

        # price = 0 (call), 1 (put) for spot > strike and volatility --> +0
        result = compute_price(c, torch.tensor([[0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([1.0]).to(device))
        result = compute_price(p, torch.tensor([[0.01, 1.0, 1e-10]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

    @pytest.mark.gpu
    def test_price_limit_gpu(self):
        self.test_price_limit(device="cuda")

    def test_price_monte_carlo(self, device: str = "cpu"):
        d = EuropeanBinaryOption(BrownianStock()).to(device)
        m = BSEuropeanBinaryOption.from_derivative(d).to(device)
        torch.manual_seed(42)
        d.simulate(n_paths=int(1e6))

        input = torch.tensor([[0.0, d.maturity, d.ul().sigma]]).to(device)
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

        d = EuropeanBinaryOption(BrownianStock(), call=False).to(device)
        m = BSEuropeanBinaryOption.from_derivative(d).to(device)
        torch.manual_seed(42)
        d.simulate(n_paths=int(1e6))

        input = torch.tensor([[0.0, d.maturity, d.ul().sigma]]).to(device)
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

    @pytest.mark.gpu
    def test_price_monte_carlo_gpu(self):
        self.test_price_monte_carlo(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_autogreek(self, call, device: str = "cpu"):
        m = BSEuropeanBinaryOption(call=call).to(device)
        s = torch.linspace(-0.5, 0.5, 10).to(device)
        t = torch.full_like(s, 1.0)
        v = torch.full_like(s, 0.2)

        result = m.delta(s, t, v)
        expect = autogreek.delta(
            m.price, log_moneyness=s, time_to_maturity=t, volatility=v, strike=1.0
        )
        assert_close(result, expect, atol=0, rtol=1e-4)

        result = m.gamma(s, t, v)
        expect = autogreek.gamma(
            m.price, log_moneyness=s, time_to_maturity=t, volatility=v, strike=1.0
        )
        assert_close(result, expect, atol=0, rtol=1e-4)

        result = m.vega(s, t, v)
        expect = autogreek.vega(
            m.price, log_moneyness=s, time_to_maturity=t, volatility=v, strike=1.0
        )
        assert_close(result, expect, atol=0, rtol=1e-4)

        result = m.theta(s, t, v)
        expect = autogreek.theta(
            m.price, log_moneyness=s, time_to_maturity=t, volatility=v, strike=1.0
        )
        assert_close(result, expect, atol=0, rtol=1e-4)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_autogreek_gpu(self, call: bool):
        self.test_autogreek(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_delta_2(self, call: bool, device: str = "cpu"):
        m = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.delta(
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.delta(
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )
        result = m.delta(
            torch.tensor(1.0).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.delta(
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.delta(
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.delta(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, float("inf") if call else -float("inf"))
        assert_close(result, expect)
        result = m.delta(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, float("inf") if call else -float("inf"))
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_delta_2_gpu(self, call: bool):
        self.test_delta_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_delta_3(self, call: bool, device: str = "cpu"):
        derivative = EuropeanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.delta(None, torch.tensor(1).to(device), torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.delta(torch.tensor(1).to(device), None, torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.delta(torch.tensor(1), torch.tensor(2), None)
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.delta()
        expect = m2.delta(
            derivative.log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.delta(
            None, derivative.time_to_maturity(), derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.delta(
            derivative.log_moneyness(), None, derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.delta(
            derivative.log_moneyness(), derivative.time_to_maturity(), None
        )
        assert_close(result, expect)
        with pytest.raises(ValueError):
            m2.delta(
                None, derivative.time_to_maturity(), derivative.underlier.volatility
            )
        with pytest.raises(ValueError):
            m2.delta(derivative.log_moneyness(), None, derivative.underlier.volatility)
        with pytest.raises(ValueError):
            m2.delta(derivative.log_moneyness(), derivative.time_to_maturity(), None)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_delta_3_gpu(self, call: bool):
        self.test_delta_3(call, device="cuda")

    def test_gamma(self, device: str = "cpu"):
        m = BSEuropeanBinaryOption().to(device)
        result = m.gamma(
            torch.tensor(0.01).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0.2).to(device),
        )
        expect = torch.tensor(-1.4645787477493286).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_gamma_gpu(self):
        self.test_gamma(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_gamma_valueerror(self, call: bool, device: str = "cpu"):
        m = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.gamma(
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.gamma(
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_gamma_valueerror_gpu(self, call: bool):
        self.test_gamma_valueerror(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_gamma_3(self, call: bool, device: str = "cpu"):
        derivative = EuropeanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.gamma(None, torch.tensor(1).to(device), torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.gamma(torch.tensor(1).to(device), None, torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.gamma(torch.tensor(1).to(device), torch.tensor(2).to(device), None)
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.gamma()
        expect = m2.gamma(
            derivative.log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        # ToDo: [..., :-1] should be removed
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.gamma(
            None, derivative.time_to_maturity(), derivative.underlier.volatility
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.gamma(
            derivative.log_moneyness(), None, derivative.underlier.volatility
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.gamma(
            derivative.log_moneyness(), derivative.time_to_maturity(), None
        )
        assert_close(result[..., :-1], expect[..., :-1])
        with pytest.raises(ValueError):
            m2.gamma(
                None, derivative.time_to_maturity(), derivative.underlier.volatility
            )
        with pytest.raises(ValueError):
            m2.gamma(derivative.log_moneyness(), None, derivative.underlier.volatility)
        with pytest.raises(ValueError):
            m2.gamma(derivative.log_moneyness(), derivative.time_to_maturity(), None)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_gamma_3_gpu(self, call: bool):
        self.test_gamma_3(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_vega_valueerror(self, call: bool, device: str = "cpu"):
        m = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.vega(
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.vega(
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_vega_valueerror_gpu(self, call: bool):
        self.test_vega_valueerror(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_vega_3(self, call: bool, device: str = "cpu"):
        derivative = EuropeanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.vega(None, torch.tensor(1).to(device), torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.vega(torch.tensor(1).to(device), None, torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.vega(torch.tensor(1).to(device), torch.tensor(2).to(device), None)
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.vega()
        expect = m2.vega(
            derivative.log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        # ToDo: [..., :-1] should be removed
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.vega(
            None, derivative.time_to_maturity(), derivative.underlier.volatility
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.vega(
            derivative.log_moneyness(), None, derivative.underlier.volatility
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.vega(derivative.log_moneyness(), derivative.time_to_maturity(), None)
        assert_close(result[..., :-1], expect[..., :-1])
        with pytest.raises(ValueError):
            m2.vega(
                None, derivative.time_to_maturity(), derivative.underlier.volatility
            )
        with pytest.raises(ValueError):
            m2.vega(derivative.log_moneyness(), None, derivative.underlier.volatility)
        with pytest.raises(ValueError):
            m2.vega(derivative.log_moneyness(), derivative.time_to_maturity(), None)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_vega_3_gpu(self, call: bool):
        self.test_vega_3(call, device="cuda")

    def test_vega_and_gamma(self, device: str = "cpu"):
        m = BSEuropeanBinaryOption().to(device)
        # vega = spot^2 * sigma * (T - t) * gamma
        # See Chapter 5 Appendix A, Bergomi "Stochastic volatility modeling"
        spot = torch.tensor([0.9, 1.0, 1.1]).to(device)
        t = torch.tensor([0.1, 0.2, 0.3]).to(device)
        v = torch.tensor([0.1, 0.2, 0.3]).to(device)
        vega = m.vega(spot.log(), t, v)
        gamma = m.gamma(spot.log(), t, v)
        assert_close(vega, spot.square() * v * t * gamma, atol=1e-3, rtol=0)

    @pytest.mark.gpu
    def test_vega_and_gamma_gpu(self):
        self.test_vega_and_gamma(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_vega_and_gamma_2(self, call: bool, device: str = "cpu"):
        derivative = EuropeanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanBinaryOption.from_derivative(derivative).to(device)
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        vega = m.vega()
        gamma = m.gamma()
        result = vega[..., :-1]
        expect = (
            derivative.underlier.spot.square()
            * derivative.underlier.volatility
            * derivative.time_to_maturity()
            * gamma
        )[..., :-1]
        assert_close(result, expect, atol=0, rtol=1e-4)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_vega_and_gamma_2_gpu(self, call: bool):
        self.test_vega_and_gamma_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_theta_2(self, call: bool, device: str = "cpu"):
        m = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.theta(
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.theta(
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_theta_2_gpu(self, call: bool):
        self.test_theta_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_theta_3(self, call: bool, device: str = "cpu"):
        derivative = EuropeanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.theta(None, torch.tensor(1).to(device), torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.theta(torch.tensor(1).to(device), None, torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.theta(torch.tensor(1).to(device), torch.tensor(2).to(device), None)
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.theta()
        expect = m2.theta(
            derivative.log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        # ToDo: [..., :-1] should be removed
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.theta(
            None, derivative.time_to_maturity(), derivative.underlier.volatility
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.theta(
            derivative.log_moneyness(), None, derivative.underlier.volatility
        )
        assert_close(result[..., :-1], expect[..., :-1])
        result = m.theta(
            derivative.log_moneyness(), derivative.time_to_maturity(), None
        )
        assert_close(result[..., :-1], expect[..., :-1])
        with pytest.raises(ValueError):
            m2.theta(
                None, derivative.time_to_maturity(), derivative.underlier.volatility
            )
        with pytest.raises(ValueError):
            m2.theta(derivative.log_moneyness(), None, derivative.underlier.volatility)
        with pytest.raises(ValueError):
            m2.theta(derivative.log_moneyness(), derivative.time_to_maturity(), None)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_theta_3_gpu(self, call: bool):
        self.test_theta_3(call, device="cuda")

    def test_price(self, device: str = "cpu"):
        m = BSEuropeanBinaryOption().to(device)

        result = m.price(
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
            torch.tensor(0.2).to(device),
        )
        expect = torch.tensor(0.4874).to(device)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

        result = m.price(
            torch.tensor(0.0001).to(device),
            torch.tensor(0.1).to(device),
            torch.tensor(0.2).to(device),
        )
        expect = torch.tensor(0.4880).to(device)
        assert_close(result, expect, atol=1e-4, rtol=1e-4)

    @pytest.mark.gpu
    def test_price_gpu(self):
        self.test_price(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_price_2(self, call: bool, device: str = "cpu"):
        m = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.price(
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.price(
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )
        result = m.price(
            torch.tensor(1.0 if call else -1.0).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0).to(device),
        )
        expect = torch.full_like(result, 1)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(1.0 if call else -1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 1)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(1.0 if call else -1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 1)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 0.5)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 0.5)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_price_2_gpu(self, call: bool):
        self.test_price_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_price_3(self, call: bool, device: str = "cpu"):
        derivative = EuropeanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.price(None, torch.tensor(1).to(device), torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.price(torch.tensor(1).to(device), None, torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.price(torch.tensor(1).to(device), torch.tensor(2).to(device), None)
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.price()
        expect = m2.price(
            derivative.log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.price(
            None, derivative.time_to_maturity(), derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.price(
            derivative.log_moneyness(), None, derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.price(
            derivative.log_moneyness(), derivative.time_to_maturity(), None
        )
        assert_close(result, expect)
        with pytest.raises(ValueError):
            m2.price(
                None, derivative.time_to_maturity(), derivative.underlier.volatility
            )
        with pytest.raises(ValueError):
            m2.price(derivative.log_moneyness(), None, derivative.underlier.volatility)
        with pytest.raises(ValueError):
            m2.price(derivative.log_moneyness(), derivative.time_to_maturity(), None)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_price_3_gpu(self, call: bool):
        self.test_price_3(call, device="cuda")

    def test_implied_volatility(self, device: str = "cpu"):
        lm = torch.full((3,), -0.01).to(device)
        t = torch.full((3,), 0.1).to(device)
        price = torch.tensor([0.40, 0.41, 0.42]).to(device)

        m = BSEuropeanBinaryOption().to(device)
        iv = m.implied_volatility(lm, t, price)

        result = BSEuropeanBinaryOption().to(device).price(lm, t, iv)
        assert_close(result, price, check_stride=False)

    @pytest.mark.gpu
    def test_implied_volatility_gpu(self):
        self.test_implied_volatility(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_implied_volatility_2(self, call: bool, device: str = "cpu"):
        derivative = EuropeanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.implied_volatility()
        with pytest.raises(AttributeError):
            m.implied_volatility(
                None, torch.tensor(1).to(device), torch.tensor(1).to(device)
            )
        with pytest.raises(AttributeError):
            m.implied_volatility(
                torch.tensor(1).to(device), None, torch.tensor(1).to(device)
            )
        with pytest.raises(ValueError):
            m.implied_volatility(
                torch.tensor(0).to(device), torch.tensor(0).to(device), None
            )
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        with pytest.raises(ValueError):
            m.implied_volatility()
        result = m.implied_volatility(price=derivative.underlier.spot)
        expect = m2.implied_volatility(
            derivative.log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.spot,
        )
        assert_close(result, expect)
        result = m.implied_volatility(
            None, derivative.time_to_maturity(), derivative.underlier.spot
        )
        assert_close(result, expect)
        result = m.implied_volatility(
            derivative.log_moneyness(), None, derivative.underlier.spot
        )
        assert_close(result, expect)
        with pytest.raises(ValueError):
            m2.implied_volatility(
                None, derivative.time_to_maturity(), derivative.underlier.spot
            )
        with pytest.raises(ValueError):
            m2.implied_volatility(
                derivative.log_moneyness(), None, derivative.underlier.spot
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_implied_volatility_2_gpu(self, call: bool):
        self.test_implied_volatility_2(call, device="cuda")

    def test_example(self, device: str = "cpu"):
        from pfhedge.instruments import BrownianStock
        from pfhedge.instruments import EuropeanBinaryOption
        from pfhedge.nn import Hedger

        derivative = EuropeanBinaryOption(BrownianStock()).to(device)
        model = BSEuropeanBinaryOption.from_derivative(derivative).to(device)
        hedger = Hedger(model, model.inputs()).to(device)
        result = hedger.price(derivative)
        expect = torch.tensor(0.4922).to(device)
        x = hedger.compute_hedge(derivative)
        assert not x.isnan().any()
        assert_close(result, expect, atol=1e-2, rtol=1e-2)

    @pytest.mark.gpu
    def test_example_gpu(self):
        self.test_example(device="cuda")

    def test_shape(self, device: str = "cpu"):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSEuropeanBinaryOption().to(device)
        self.assert_shape_delta(m, device=device)
        self.assert_shape_gamma(m, device=device)
        self.assert_shape_vega(m, device=device)
        self.assert_shape_theta(m, device=device)
        self.assert_shape_price(m, device=device)
        self.assert_shape_forward(m, device=device)

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")
