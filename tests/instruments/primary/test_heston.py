import pytest
import torch

from pfhedge.instruments import HestonStock


class TestHestonStock:
    @pytest.mark.parametrize("seed", range(1))
    def test_values_are_finite(self, seed):
        torch.manual_seed(seed)

        s = HestonStock()
        s.simulate(n_paths=1000)

        print(s)
        assert not s.variance.isnan().any()

    def test_repr(self):
        s = HestonStock(cost=1e-4)
        expect = (
            "HestonStock(kappa=1.00e+00, theta=4.00e-02, sigma=2.00e-01, "
            "rho=-7.00e-01, cost=1.00e-04, dt=4.00e-03)"
        )
        assert repr(s) == expect
