from math import sqrt

import pytest
import torch
from torch.testing import assert_close

from pfhedge.features._getter import get_feature
from pfhedge.instruments import AmericanBinaryOption
from pfhedge.instruments import BrownianStock
from pfhedge.nn import BSAmericanBinaryOption

from ._base import _TestBSModule
from ._utils import compute_delta
from ._utils import compute_gamma
from ._utils import compute_price
from ._utils import compute_theta
from ._utils import compute_vega


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

    def test_check_delta(self, device: str = "cpu"):
        m = BSAmericanBinaryOption().to(device)

        # delta = 0 for max --> +0
        result = compute_delta(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # delta = 0 for max > 0
        result = compute_delta(m, torch.tensor([[0.0, 0.01, 0.1, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # delta = 0 for time --> +0
        result = compute_delta(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # delta = 0 for volatility --> +0
        result = compute_delta(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_check_delta_gpu(self):
        self.test_check_delta(device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_delta_2(self, call: bool, device: str = "cpu"):
        m = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.delta(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.delta(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_delta_2_gpu(self, call: bool):
        self.test_delta_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_delta_3(self, call: bool, device: str = "cpu"):
        derivative = AmericanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSAmericanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.delta()
        with pytest.raises(AttributeError):
            m.delta(
                None,
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.delta(
                torch.tensor(1).to(device),
                None,
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.delta(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                None,
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.delta(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
                None,
            )
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.delta()
        expect = m2.delta(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.delta(
            None,
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.delta(
            derivative.log_moneyness(),
            None,
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.delta(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            None,
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.delta(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            None,
        )
        assert_close(result, expect)
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_delta_3_gpu(self, call: bool):
        self.test_delta_3(call, device="cuda")

    def test_check_gamma(self, device: str = "cpu"):
        m = BSAmericanBinaryOption().to(device)

        # gamma = 0 for max --> +0
        result = compute_gamma(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # gamma = 0 for max > 0
        result = compute_gamma(m, torch.tensor([[0.0, 0.01, 0.1, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # gamma = 0 for time --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # gamma = 0 for volatility --> +0
        result = compute_gamma(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_check_gamma_gpu(self):
        self.test_check_gamma(device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_gamma_2(self, call: bool, device: str = "cpu"):
        m = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.gamma(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.gamma(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_gamma_2_gpu(self, call: bool):
        self.test_gamma_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_gamma_3(self, call: bool, device: str = "cpu"):
        derivative = AmericanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSAmericanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.gamma(
                None,
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.gamma(
                torch.tensor(1).to(device),
                None,
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.gamma(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                None,
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.gamma(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
                None,
            )
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_gamma_3_gpu(self, call: bool):
        self.test_gamma_3(call, device="cuda")

    def test_check_vega(self, device: str = "cpu"):
        m = BSAmericanBinaryOption().to(device)

        # vega = 0 for max --> +0
        result = compute_vega(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # vega = 0 for max > 0
        result = compute_vega(m, torch.tensor([[0.0, 0.01, 0.1, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # vega = 0 for time --> +0
        result = compute_vega(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # vega = 0 for volatility --> +0
        result = compute_vega(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_check_vega_gpu(self):
        self.test_check_vega(device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_vega_2(self, call: bool, device: str = "cpu"):
        m = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.vega(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.vega(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_vega_2_gpu(self, call: bool):
        self.test_vega_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_vega_3(self, call: bool, device: str = "cpu"):
        derivative = AmericanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSAmericanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.vega(
                None,
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.vega(
                torch.tensor(1).to(device),
                None,
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.vega(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                None,
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.vega(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
                None,
            )
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_vega_3_gpu(self, call: bool):
        self.test_vega_3(call, device="cuda")

    def test_check_theta(self, device: str = "cpu"):
        m = BSAmericanBinaryOption().to(device)

        # vega = 0 for max --> +0
        result = compute_theta(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # vega = 0 for max > 0
        result = compute_theta(m, torch.tensor([[0.0, 0.01, 0.1, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # vega = 0 for time --> +0
        result = compute_theta(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # vega = 0 for volatility --> +0
        result = compute_theta(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_check_theta_gpu(self):
        self.test_check_theta(device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_theta_2(self, call: bool, device: str = "cpu"):
        m = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.theta(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.theta(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_theta_2_gpu(self, call: bool):
        self.test_theta_2(call, device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_theta_3(self, call: bool, device: str = "cpu"):
        derivative = AmericanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSAmericanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.theta(
                None,
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.theta(
                torch.tensor(1).to(device),
                None,
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.theta(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                None,
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.theta(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
                None,
            )
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_theta_3_gpu(self, call: bool):
        self.test_theta_3(call, device="cuda")

    def test_check_price(self, device: str = "cpu"):
        m = BSAmericanBinaryOption().to(device)

        # price = 0 for max --> +0
        result = compute_price(m, torch.tensor([[-10.0, -10.0, 0.1, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # price = 1 for max > 0
        result = compute_price(m, torch.tensor([[0.0, 0.01, 0.1, 0.2]]).to(device))
        expect = torch.tensor([1.0]).to(device)
        assert_close(result, expect)

        # price = 0 for time --> +0
        result = compute_price(m, torch.tensor([[-0.01, -0.01, 1e-10, 0.2]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

        # price = 0 for volatility --> +0
        result = compute_price(m, torch.tensor([[-0.01, -0.01, 0.1, 1e-10]]).to(device))
        expect = torch.tensor([0.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_check_price_gpu(self):
        self.test_check_price(device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_price_3(self, call: bool, device: str = "cpu"):
        m = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(ValueError):
            m.price(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(-1.0).to(device),
                torch.tensor(0.2).to(device),
            )
        with pytest.raises(ValueError):
            m.price(
                torch.tensor(0.0).to(device),
                torch.tensor(0.0).to(device),
                torch.tensor(1.0).to(device),
                torch.tensor(-0.2).to(device),
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_price_3_gpu(self, call: bool):
        self.test_price_3(call, device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_price_4(self, call: bool, device: str = "cpu"):
        derivative = AmericanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSAmericanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.price(
                None,
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.price(
                torch.tensor(1).to(device),
                None,
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.price(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                None,
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.price(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
                None,
            )
        torch.manual_seed(42)
        derivative.simulate(n_paths=1)
        result = m.price()
        expect = m2.price(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.price(
            None,
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.price(
            derivative.log_moneyness(),
            None,
            derivative.time_to_maturity(),
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.price(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            None,
            derivative.underlier.volatility,
        )
        assert_close(result, expect)
        result = m.price(
            derivative.log_moneyness(),
            derivative.max_log_moneyness(),
            derivative.time_to_maturity(),
            None,
        )
        assert_close(result, expect)
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_price_4_gpu(self, call: bool):
        self.test_price_4(call, device="cuda")

    def test_check_price_monte_carlo(self, device: str = "cpu"):
        torch.manual_seed(42)

        # Monte Carlo evaluation of a lookback option needs small dt
        k = 1.01
        d = AmericanBinaryOption(BrownianStock(dt=1e-5), strike=k).to(device)
        m = BSAmericanBinaryOption.from_derivative(d)
        d.simulate(n_paths=int(1e4), init_state=(1.0,))

        s = torch.tensor([1.0 / k]).to(device).log()
        input = torch.tensor([[s, s, d.maturity, d.ul().sigma]]).to(device)
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

        # Continuity correction according to:
        # Broadie, M., Glasserman, P. and Kou, S., 1997.
        # A continuity correction for discrete barrier options.
        # Mathematical Finance, 7(4), pp.325-349.
        beta = 0.5825971579  # -zeta(1/2) / sqrt(2 pi)
        k = 1.01
        d = AmericanBinaryOption(BrownianStock(), strike=k).to(device)
        m = BSAmericanBinaryOption.from_derivative(d)
        d.simulate(n_paths=int(1e5), init_state=(1.0,))

        k_shift = (
            k * torch.tensor(beta * d.ul().sigma * sqrt(d.ul().dt)).to(device).exp()
        )

        s = torch.tensor([1.0 / k_shift]).to(device).log()
        input = torch.tensor([[s, s, d.maturity, d.ul().sigma]]).to(device)
        result = compute_price(m, input)
        expect = d.payoff().mean(0, keepdim=True)
        assert_close(result, expect, rtol=1e-2, atol=0.0)

    @pytest.mark.gpu
    def test_check_price_monte_carlo_gpu(self):
        self.test_check_price_monte_carlo(device="cuda")

    def test_vega_and_gamma(self, device: str = "cpu"):
        m = BSAmericanBinaryOption().to(device)
        # vega = spot^2 * sigma * (T - t) * gamma
        # See Chapter 5 Appendix A, Bergomi "Stochastic volatility modeling"
        spot = torch.tensor([0.90, 0.94, 0.98]).to(device)
        t = torch.tensor([0.1, 0.2, 0.3]).to(device)
        v = torch.tensor([0.1, 0.2, 0.3]).to(device)
        vega = m.vega(spot.log(), spot.log(), t, v)
        gamma = m.gamma(spot.log(), spot.log(), t, v)
        assert_close(vega, spot.square() * v * t * gamma, atol=1e-3, rtol=0)

    @pytest.mark.gpu
    def test_vega_and_gamma_gpu(self):
        self.test_vega_and_gamma(device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_vega_and_gamma_2(self, call: bool, device: str = "cpu"):
        derivative = AmericanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSAmericanBinaryOption.from_derivative(derivative).to(device)
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_vega_and_gamma_2_gpu(self, call: bool):
        self.test_vega_and_gamma_2(call, device="cuda")

    def test_features(self):
        m = BSAmericanBinaryOption()
        assert m.inputs() == [
            "log_moneyness",
            "max_log_moneyness",
            "time_to_maturity",
            "volatility",
        ]
        _ = [get_feature(f) for f in m.inputs()]

    def test_implied_volatility(self, device: str = "cpu"):
        # log_moneyness, max_log_moneyness, time_to_maturity, price
        input = torch.tensor(
            [
                [-0.01, -0.01, 0.1, 0.5],
                [-0.01, -0.01, 0.1, 0.6],
                [-0.01, -0.01, 0.1, 0.7],
            ]
        ).to(device)
        m = BSAmericanBinaryOption().to(device)
        iv = m.implied_volatility(input[:, 0], input[:, 1], input[:, 2], input[:, 3])

        result = BSAmericanBinaryOption().price(
            input[:, 0], input[:, 1], input[:, 2], iv
        )
        expect = input[:, -1]
        assert_close(result, expect, atol=1e-4, rtol=1e-4, check_stride=False)

    @pytest.mark.gpu
    def test_implied_volatility_gpu(self):
        self.test_implied_volatility(device="cuda")

    @pytest.mark.parametrize("call", [True])
    def test_implied_volatility_2(self, call: bool, device: str = "cpu"):
        derivative = AmericanBinaryOption(BrownianStock(), call=call).to(device)
        m = BSAmericanBinaryOption.from_derivative(derivative).to(device)
        m2 = BSAmericanBinaryOption(call=call).to(device)
        with pytest.raises(AttributeError):
            m.implied_volatility(
                None,
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.implied_volatility(
                torch.tensor(1).to(device),
                None,
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
            )
        with pytest.raises(AttributeError):
            m.implied_volatility(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                None,
                torch.tensor(3).to(device),
            )
        with pytest.raises(ValueError):
            m.implied_volatility(
                torch.tensor(1).to(device),
                torch.tensor(2).to(device),
                torch.tensor(3).to(device),
                None,
            )
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

    @pytest.mark.gpu
    @pytest.mark.parametrize("call", [True])
    def test_implied_volatility_2_gpu(self, call: bool):
        self.test_implied_volatility_2(call, device="cuda")

    def test_shape(self, device: str = "cpu"):
        torch.distributions.Distribution.set_default_validate_args(False)

        m = BSAmericanBinaryOption().to(device)
        self.assert_shape_delta(m, device=device)
        self.assert_shape_gamma(m, device=device)
        self.assert_shape_vega(m, device=device)
        self.assert_shape_theta(m, device=device)
        self.assert_shape_price(m, device=device)
        self.assert_shape_forward(m, device=device)

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")
