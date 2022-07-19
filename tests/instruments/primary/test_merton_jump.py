import pytest
import torch

from pfhedge.instruments import MertonJumpStock


class TestMertonJumpStock:
    @pytest.mark.parametrize("seed", range(1))
    def test_values_are_finite(self, seed):
        torch.manual_seed(seed)

        s = MertonJumpStock()
        s.simulate(n_paths=1000)

        assert not s.variance.isnan().any()

    def test_repr(self):
        s = MertonJumpStock(cost=1e-4)
        expect = "MertonJumpStock(\
mu=0., sigma=0.2000, jump_per_year=68, jump_mean=0., jump_std=0.0100, cost=1.0000e-04, dt=0.0040)"
        assert repr(s) == expect

    def test_simulate_shape(self):
        s = MertonJumpStock(dt=0.1)
        s.simulate(time_horizon=0.2, n_paths=10)
        assert s.spot.size() == torch.Size((10, 3))
        assert s.variance.size() == torch.Size((10, 3))

        s = MertonJumpStock(dt=0.1)
        s.simulate(time_horizon=0.25, n_paths=10)
        assert s.spot.size() == torch.Size((10, 4))
        assert s.variance.size() == torch.Size((10, 4))
