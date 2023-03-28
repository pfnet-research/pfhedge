import pytest
import torch
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BSEuropeanOption

from ._base import _TestBSModule
from ._utils import compute_delta
from ._utils import compute_gamma
from ._utils import compute_price


class TestBSEuropeanOption(_TestBSModule):
    def test_repr(self):
        m = BSEuropeanOption()
        assert repr(m) == "BSEuropeanOption(strike=1.)"

        derivative = EuropeanOption(BrownianStock(), strike=1.1, call=False)
        m = BSEuropeanOption.from_derivative(derivative)
        assert repr(m) == "BSEuropeanOption(call=False, strike=1.1000)"

    def test_features(self):
        m = BSEuropeanOption()
        assert m.inputs() == ["log_moneyness", "time_to_maturity", "volatility"]
        _ = [get_feature(f) for f in m.inputs()]

    def test_delta_limit(self, device: str = "cpu"):
        EPSILON = 1e-10
        c = BSEuropeanOption().to(device)
        p = BSEuropeanOption(call=False).to(device)

        # delta = 0 (call), -1 (put) for spot --> +0
        result = compute_delta(c, torch.tensor([[-10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_delta(p, torch.tensor([[-10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([-1.0]).to(device))

        # delta = 1 (call), 0 (put) for spot --> +inf
        result = compute_delta(c, torch.tensor([[10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([1.0]).to(device))
        result = compute_delta(p, torch.tensor([[10.0, 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # delta = 0 (call), -1 (put) for spot < k and time --> +0
        result = compute_delta(c, torch.tensor([[-0.01, EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_delta(p, torch.tensor([[-0.01, EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([-1.0]).to(device))

        # delta = 1 (call), 0 (put) for spot > k and time --> +0
        result = compute_delta(c, torch.tensor([[0.01, EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([1.0]).to(device))
        result = compute_delta(p, torch.tensor([[0.01, EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # delta = 0 (call), -1 (put) for spot < k and volatility --> +0
        result = compute_delta(c, torch.tensor([[-0.01, 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_delta(p, torch.tensor([[-0.01, 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([-1.0]).to(device))

        # delta = 1 (call), 0 (put) for spot > k and volatility --> +0
        result = compute_delta(c, torch.tensor([[0.01, 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([1.0]).to(device))
        result = compute_delta(p, torch.tensor([[0.01, 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

    @pytest.mark.gpu
    def test_delta_limit_gpu(self):
        self.test_delta_limit(device="cuda")

    def test_gamma_limit(self, device: str = "cpu"):
        EPSILON = 1e-10
        c = BSEuropeanOption().to(device)
        p = BSEuropeanOption(call=False).to(device)

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

        # gamma = 0 for spot / k < 1 and time --> +0
        result = compute_gamma(c, torch.tensor([[-0.01, EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[-0.01, EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # gamma = 0 for spot / k > 1 and time --> +0
        result = compute_gamma(c, torch.tensor([[0.01, EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[0.01, EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # gamma = 0 for spot / k < 1 and volatility --> +0
        result = compute_gamma(c, torch.tensor([[-0.01, 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[-0.01, 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # gamma = 0 for spot / k > 1 and volatility --> +0
        result = compute_gamma(c, torch.tensor([[0.01, 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_gamma(p, torch.tensor([[0.01, 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

    @pytest.mark.gpu
    def test_gamma_limit_gpu(self):
        self.test_gamma_limit(device="cuda")

    def test_price_limit(self, device: str = "cpu"):
        EPSILON = 1e-10
        c = BSEuropeanOption().to(device)
        p = BSEuropeanOption(call=False).to(device)

        # price = 0 (call), k - spot (put) for spot --> +0
        s = torch.tensor([-10.0]).to(device).exp()
        result = compute_price(c, torch.tensor([[s.log(), 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_price(p, torch.tensor([[s.log(), 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([1.0 - s]).to(device))

        # price = spot - k (call), 0 (put) for spot --> +inf
        s = torch.tensor([10.0]).to(device).exp()
        result = compute_price(c, torch.tensor([[s.log(), 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([s - 1.0]).to(device))
        result = compute_price(p, torch.tensor([[s.log(), 1.0, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # price = 0 (call), k - s (put) for spot < k and time --> +0
        s = torch.tensor([-0.01]).to(device).exp()
        result = compute_price(c, torch.tensor([[s.log(), EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_price(p, torch.tensor([[s.log(), EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([1.0 - s]).to(device))

        # price = spot - k (call), 0 (put) for spot > k and time --> +0
        s = torch.tensor([0.01]).to(device).exp()
        result = compute_price(c, torch.tensor([[s.log(), EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([s - 1.0]).to(device))
        result = compute_price(p, torch.tensor([[s.log(), EPSILON, 0.2]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

        # price = 0 (call), k - spot (put) for spot < k and volatility --> +0
        s = torch.tensor([-0.01]).to(device).exp()
        result = compute_price(c, torch.tensor([[s.log(), 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))
        result = compute_price(p, torch.tensor([[s.log(), 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([1.0 - s]).to(device))

        # price = spot - k (call), 0 (put) for spot > k and volatility --> +0
        s = torch.tensor([0.01]).to(device).exp()
        result = compute_price(c, torch.tensor([[s.log(), 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([s - 1.0]).to(device))
        result = compute_price(p, torch.tensor([[s.log(), 1.0, EPSILON]]).to(device))
        assert_close(result, torch.tensor([0.0]).to(device))

    @pytest.mark.gpu
    def test_price_limit_gpu(self):
        self.test_price_limit(device="cuda")

    def test_price_monte_carlo(self, device: str = "cpu"):
        d = EuropeanOption(BrownianStock()).to(device)
        m = BSEuropeanOption.from_derivative(d).to(device)
        torch.manual_seed(42)
        d.simulate(n_paths=int(1e6))

        input = torch.tensor([[0.0, d.maturity, d.ul().sigma]]).to(device)
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        print(result, expect)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

        d = EuropeanOption(BrownianStock(), call=False).to(device)
        m = BSEuropeanOption.from_derivative(d).to(device)
        torch.manual_seed(42)
        d.simulate(n_paths=int(1e6))

        input = torch.tensor([[0.0, d.maturity, d.ul().sigma]]).to(device)
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        print(result, expect)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

    @pytest.mark.gpu
    def test_price_monte_carlo_gpu(self):
        self.test_price_monte_carlo(device="cuda")

    def test_forward_2(self, device: str = "cpu"):
        m = BSEuropeanOption(call=False).to(device)
        input = torch.tensor([[0.0, 1.0, 0.2]]).to(device)
        result = m(input)
        expect = torch.full_like(result, -0.4601721)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_forward_2_gpu(self):
        self.test_forward_2(device="cuda")

    def test_forward_3(self, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), call=False).to(device)
        m = BSEuropeanOption.from_derivative(derivative).to(device)
        input = torch.tensor([[0.0, 1.0, 0.2]]).to(device)
        result = m(input)
        expect = torch.full_like(result, -0.4601721)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_forward_3_gpu(self):
        self.test_forward_3(device="cuda")

    def test_delta_1(self, device: str = "cpu"):
        m = BSEuropeanOption().to(device)
        result = m.delta(
            torch.tensor(0.0).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0.2).to(device),
        )
        expect = torch.full_like(result, 0.5398278962)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_delta_1_gpu(self):
        self.test_delta_1(device="cuda")

    def test_delta_2(self, device: str = "cpu"):
        m = BSEuropeanOption(call=False).to(device)
        result = m.delta(
            torch.tensor(0.0).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0.2).to(device),
        )
        expect = torch.full_like(result, -0.4601721)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_delta_2_gpu(self):
        self.test_delta_2(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_delta_3(self, call: bool, device: str = "cpu"):
        m = BSEuropeanOption(call=call).to(device)
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
        expect = torch.full_like(result, 1.0 if call else 0.0)
        assert_close(result, expect)
        result = m.delta(
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 1.0 if call else 0.0)
        assert_close(result, expect)
        result = m.delta(
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 1.0 if call else 0.0)
        assert_close(result, expect)
        result = m.delta(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 0.5 if call else -0.5)
        assert_close(result, expect)
        result = m.delta(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 0.5 if call else -0.5)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_delta_3_gpu(self, call: bool):
        self.test_delta_3(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_delta_4(self, call: bool, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.delta(None, torch.tensor(1).to(device), torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.delta(torch.tensor(1).to(device), None, torch.tensor(2).to(device))
        with pytest.raises(AttributeError):
            m.delta(torch.tensor(1).to(device), torch.tensor(2).to(device), None)
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
    def test_delta_4_gpu(self, call: bool):
        self.test_delta_4(call, device="cuda")

    def test_gamma_1(self, device: str = "cpu"):
        m = BSEuropeanOption().to(device)
        result = m.gamma(
            torch.tensor(0.0).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0.2).to(device),
        )
        expect = torch.full_like(result, 1.9847627374)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_gamma_1_gpu(self):
        self.test_gamma_1(device="cuda")

    def test_gamma_2(self, device: str = "cpu"):
        m = BSEuropeanOption(call=False).to(device)
        result = m.gamma(
            torch.tensor(0.0).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0.2).to(device),
        )
        expect = torch.full_like(result, 1.9847627374)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_gamma_2_gpu(self):
        self.test_gamma_2(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_gamma_3(self, call: bool, device: str = "cpu"):
        m = BSEuropeanOption(call=call).to(device)
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
        result = m.gamma(
            torch.tensor(1.0).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.gamma(
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.gamma(
            torch.tensor(1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.gamma(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, float("inf"))
        assert_close(result, expect)
        result = m.gamma(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, float("inf"))
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_gamma_3_gpu(self, call: bool):
        self.test_gamma_3(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_gamma_4(self, call: bool, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanOption(call=call).to(device)
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
        assert_close(result, expect)
        result = m.gamma(
            None, derivative.time_to_maturity(), derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.gamma(
            derivative.log_moneyness(), None, derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.gamma(
            derivative.log_moneyness(), derivative.time_to_maturity(), None
        )
        assert_close(result, expect)
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
    def test_gamma_4_gpu(self, call: bool):
        self.test_gamma_4(call, device="cuda")

    def test_price_1(self, device: str = "cpu"):
        m = BSEuropeanOption().to(device)
        result = m.price(
            torch.tensor(0.0).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0.2).to(device),
        )
        expect = torch.full_like(result, 0.0796557924)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_price_1_gpu(self):
        self.test_price_1(device="cuda")

    def test_price_2(self, device: str = "cpu"):
        m = BSEuropeanOption(call=False)
        result = m.price(
            torch.tensor(0.0).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0.2).to(device),
        )
        expect = torch.full_like(result, 0.0796557924)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_price_2_gpu(self):
        self.test_price_2(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_price_3(self, call: bool, device: str = "cpu"):
        m = BSEuropeanOption(call=call).to(device)
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
        expect = torch.full_like(result, 1.718282 if call else 0.632121)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(1.0 if call else -1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 1.718282 if call else 0.632121)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(1.0 if call else -1.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 1.718282 if call else 0.632121)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.price(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_price_3_gpu(self, call: bool):
        self.test_price_3(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_price_4(self, call: bool, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanOption(call=call).to(device)
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
    def test_price_4_gpu(self, call: bool):
        self.test_price_4(call, device="cuda")

    def test_implied_volatility(self, device: str = "cpu"):
        input = torch.tensor([[0.0, 0.1, 0.01], [0.0, 0.1, 0.02], [0.0, 0.1, 0.03]]).to(
            device
        )
        m = BSEuropeanOption().to(device)
        iv = m.implied_volatility(input[:, 0], input[:, 1], input[:, 2])

        result = BSEuropeanOption().to(device).price(input[:, 0], input[:, 1], iv)
        expect = input[:, 2]
        assert_close(result, expect, check_stride=False)

    @pytest.mark.gpu
    def test_implied_volatility_gpu(self):
        self.test_implied_volatility(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_implied_volatility_2(self, call: bool, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.implied_volatility(
                None, torch.tensor(1).to(device), torch.tensor(2).to(device)
            )
        with pytest.raises(AttributeError):
            m.implied_volatility(
                torch.tensor(1).to(device), None, torch.tensor(2).to(device)
            )
        with pytest.raises(ValueError):
            m.implied_volatility(
                torch.tensor(1).to(device), torch.tensor(2).to(device), None
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

    def test_vega(self, device: str = "cpu"):
        input = torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.2, 0.2], [0.0, 0.3, 0.2]]).to(
            device
        )
        m = BSEuropeanOption().to(device)
        result = m.vega(
            log_moneyness=input[..., 0],
            time_to_maturity=input[..., 1],
            volatility=input[..., 2],
        )
        expect = torch.tensor([0.1261, 0.1782, 0.2182]).to(device)
        assert_close(result, expect, atol=1e-3, rtol=0)

    @pytest.mark.gpu
    def test_vega_gpu(self):
        self.test_vega(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_vega_2(self, call: bool, device: str = "cpu"):
        m = BSEuropeanOption(call=call).to(device)
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
        result = m.vega(
            torch.tensor(0.1).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.vega(
            torch.tensor(0.1).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.vega(
            torch.tensor(0.1).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.vega(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)
        result = m.vega(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, 0.0)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_vega_2_gpu(self, call: bool):
        self.test_vega_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_vega_3(self, call: bool, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanOption(call=call).to(device)
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
        assert_close(result, expect)
        result = m.vega(
            None, derivative.time_to_maturity(), derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.vega(
            derivative.log_moneyness(), None, derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.vega(derivative.log_moneyness(), derivative.time_to_maturity(), None)
        assert_close(result, expect)
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
        m = BSEuropeanOption().to(device)
        # vega = spot^2 * sigma * (T - t) * gamma
        # See Chapter 5 Appendix A, Bergomi "Stochastic volatility modeling"
        spot = torch.tensor([0.9, 1.0, 1.1, 1.1, 1.1, 1.1]).to(device)
        t = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.0, 0.1]).to(device)
        v = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.2, 0.0]).to(device)
        vega = m.vega(spot.log(), t, v)
        gamma = m.gamma(spot.log(), t, v)
        assert_close(vega, spot.square() * v * t * gamma, atol=1e-3, rtol=0)

    @pytest.mark.gpu
    def test_vega_and_gamma_gpu(self):
        self.test_vega_and_gamma(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_vega_and_gamma_2(self, call: bool, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanOption.from_derivative(derivative).to(device)
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        vega = m.vega()
        gamma = m.gamma()
        assert_close(
            vega,
            derivative.underlier.spot.square()
            * derivative.underlier.volatility
            * derivative.time_to_maturity()
            * gamma,
            atol=1e-3,
            rtol=0,
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_vega_and_gamma_2_gpu(self, call: bool):
        self.test_vega_and_gamma_2(call, device="cuda")

    def test_theta(self, device: str = "cpu"):
        input = torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.2, 0.2], [0.0, 0.3, 0.2]]).to(
            device
        )
        m = BSEuropeanOption(strike=100)
        result = m.theta(
            log_moneyness=input[..., 0],
            time_to_maturity=input[..., 1],
            volatility=input[..., 2],
        )
        expect = torch.tensor([-12.6094, -8.9117, -7.2727]).to(device)
        assert_close(result, expect, atol=1e-3, rtol=0)

    @pytest.mark.gpu
    def test_theta_gpu(self):
        self.test_theta(device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_theta_2(self, call: bool, device: str = "cpu"):
        m = BSEuropeanOption(call=call).to(device)
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
        result = m.theta(
            torch.tensor(0.1).to(device),
            torch.tensor(1.0).to(device),
            torch.tensor(0).to(device),
        )
        expect = torch.full_like(result, -0.0)
        assert_close(result, expect)
        result = m.theta(
            torch.tensor(0.1).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, -0.0)
        assert_close(result, expect)
        result = m.theta(
            torch.tensor(0.1).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, -0.0)
        assert_close(result, expect)
        result = m.theta(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.1).to(device),
        )
        expect = torch.full_like(result, -float("inf"))
        assert_close(result, expect)
        result = m.theta(
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
        )
        expect = torch.full_like(result, -0.0)
        assert_close(result, expect)

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True, False])
    def test_theta_2_gpu(self, call: bool):
        self.test_theta_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True, False])
    def test_theta_3(self, call: bool, device: str = "cpu"):
        derivative = EuropeanOption(BrownianStock(), call=call).to(device)
        m = BSEuropeanOption.from_derivative(derivative).to(device)
        m2 = BSEuropeanOption(call=call).to(device)
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
        assert_close(result, expect)
        result = m.theta(
            None, derivative.time_to_maturity(), derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.theta(
            derivative.log_moneyness(), None, derivative.underlier.volatility
        )
        assert_close(result, expect)
        result = m.theta(
            derivative.log_moneyness(), derivative.time_to_maturity(), None
        )
        assert_close(result, expect)
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

    def test_example(self, device: str = "cpu"):
        from pfhedge.instruments import BrownianStock
        from pfhedge.instruments import EuropeanOption
        from pfhedge.nn import Hedger

        derivative = EuropeanOption(BrownianStock()).to(device)
        model = BSEuropeanOption().to(device)
        hedger = Hedger(model, model.inputs()).to(device)
        result = hedger.price(derivative)
        expect = torch.tensor(0.022).to(device)
        assert_close(result, expect, atol=1e-3, rtol=1e-3)

    @pytest.mark.gpu
    def test_example_gpu(self):
        self.test_example(device="cuda")

    def test_shape(self, device: str = "cpu"):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSEuropeanOption().to(device)
        self.assert_shape_delta(m, device=device)
        self.assert_shape_gamma(m, device=device)
        self.assert_shape_vega(m, device=device)
        self.assert_shape_theta(m, device=device)
        self.assert_shape_price(m, device=device)
        self.assert_shape_forward(m, device=device)

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")
