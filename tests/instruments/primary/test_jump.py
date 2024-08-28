import pytest
import torch

from pfhedge.instruments import KouJumpStock
from pfhedge.instruments import MertonJumpStock


class TestJumpStock:
    cls = MertonJumpStock  # Defaults to MertonJumpStock

    def setup_method(self):
        self.jump_test_class = self.cls

    @pytest.mark.parametrize("seed", range(1))
    def test_values_are_finite(self, seed, device: str = "cpu"):
        torch.manual_seed(seed)

        s = self.jump_test_class().to(device)
        s.simulate(n_paths=1000)

        assert not s.variance.isnan().any()

    @pytest.mark.gpu
    @pytest.mark.parametrize("seed", range(1))
    def test_values_are_finite_gpu(self, seed):
        self.test_values_are_finite(seed, device="cuda")

    def test_repr(self):
        s = self.jump_test_class(cost=1e-4)
        # default for merton model
        expect = "MertonJumpStock(\
mu=0., sigma=0.2000, jump_per_year=68, jump_mean=0., jump_std=0.0100, cost=1.0000e-04, dt=0.0040)"
        if self.jump_test_class == KouJumpStock:
            expect = "KouJumpStock(\
sigma=0.2000, mu=0., cost=1.0000e-04, dt=0.0040, jump_per_year=68., jump_mean_up=0.0200, jump_mean_down=0.0500, jump_up_prob=0.5000)"
        assert repr(s) == expect

    def test_simulate_shape(self, device: str = "cpu"):
        s = self.jump_test_class(dt=0.1).to(device)
        s.simulate(time_horizon=0.2, n_paths=10)
        assert s.spot.size() == torch.Size((10, 3))
        assert s.variance.size() == torch.Size((10, 3))

        s = self.jump_test_class(dt=0.1).to(device)
        s.simulate(time_horizon=0.25, n_paths=10)
        assert s.spot.size() == torch.Size((10, 4))
        assert s.variance.size() == torch.Size((10, 4))

    @pytest.mark.gpu
    def test_simulate_shape_gpu(self):
        self.test_simulate_shape(device="cuda")


class TestMertonJumpStock(TestJumpStock):
    pass  # default test checks merton's tests


class TestKouJumpStock(TestJumpStock):
    cls = KouJumpStock
