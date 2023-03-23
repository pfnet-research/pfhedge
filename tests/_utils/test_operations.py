import pytest
import torch

from pfhedge._utils.operations import ensemble_mean


class F:
    def __init__(self, device: str = "cpu"):
        self.value = 0.0
        self.device = device

    def f(self) -> float:
        self.value += 1.0
        return torch.tensor(self.value).to(self.device)


def test_ensemble_mean(device: str = "cpu"):
    f = F(device=device)
    result = ensemble_mean(f.f, n_times=10)
    expect = torch.tensor(5.5).to(device)
    assert result == expect


@pytest.mark.gpu
def test_ensemble_mean_gpu():
    test_ensemble_mean(device="cuda")
