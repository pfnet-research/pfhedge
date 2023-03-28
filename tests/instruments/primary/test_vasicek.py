import pytest
import torch

from pfhedge.instruments import VasicekRate


class TestVasicekRate:
    @pytest.mark.parametrize("seed", range(1))
    def test_values_are_finite(self, seed, device: str = "cpu"):
        torch.manual_seed(seed)

        s = VasicekRate().to(device)
        s.simulate(n_paths=1000)

        assert not s.spot.isnan().any()

    @pytest.mark.gpu
    @pytest.mark.parametrize("seed", range(1))
    def test_values_are_finite_gpu(self, seed):
        self.test_values_are_finite(seed, device="cuda")

    def test_repr(self):
        s = VasicekRate(cost=1e-4)
        expect = """\
VasicekRate(kappa=1., theta=0.0400, sigma=0.0400, cost=1.0000e-04, dt=0.0040)"""
        assert repr(s) == expect

    def test_simulate_shape(self, device: str = "cpu"):
        s = VasicekRate(dt=0.1).to(device)
        s.simulate(time_horizon=0.2, n_paths=10)
        assert s.spot.size() == torch.Size((10, 3))

        s = VasicekRate(dt=0.1).to(device)
        s.simulate(time_horizon=0.25, n_paths=10)
        assert s.spot.size() == torch.Size((10, 4))

    @pytest.mark.gpu
    def test_simulate_shape_gpu(self):
        self.test_simulate_shape(device="cuda")
