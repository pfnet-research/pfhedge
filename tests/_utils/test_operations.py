import torch

from pfhedge._utils.operations import ensemble_mean


class F:
    def __init__(self):
        self.value = 0.0

    def f(self) -> float:
        self.value += 1.0
        return torch.tensor(self.value)


def test_ensemble_mean():
    f = F()
    result = ensemble_mean(f.f, n_times=10)
    expect = torch.tensor(5.5)
    assert result == expect
