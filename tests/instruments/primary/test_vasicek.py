import pytest
import torch

from pfhedge.instruments import VasicekRate


class TestVasicekRate:
    @pytest.mark.parametrize("seed", range(1))
    def test_values_are_finite(self, seed):
        torch.manual_seed(seed)

        s = VasicekRate()
        s.simulate(n_paths=1000)

        assert not s.spot.isnan().any()

    def test_repr(self):
        s = VasicekRate(cost=1e-4)
        expect = """\
VasicekRate(kappa=1., theta=0.0400, sigma=0.0400, cost=1.0000e-04, dt=0.0040)"""
        assert repr(s) == expect

    def test_simulate_shape(self):
        s = VasicekRate(dt=0.1)
        s.simulate(time_horizon=0.2, n_paths=10)
        assert s.spot.size() == torch.Size((10, 3))

        s = VasicekRate(dt=0.1)
        s.simulate(time_horizon=0.25, n_paths=10)
        assert s.spot.size() == torch.Size((10, 4))
