import pytest
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Sequential

from pfhedge._utils.lazy import has_lazy


def test_has_lazy(device: str = "cpu"):
    m = LazyLinear(1).to(device)
    assert has_lazy(m)

    m = Linear(1, 1).to(device)
    assert not has_lazy(m)

    m = Sequential(Linear(1, 1).to(device), Linear(1, 1).to(device))
    assert not has_lazy(m)

    m = Sequential(Linear(1, 1).to(device), LazyLinear(1).to(device))
    assert has_lazy(m)


@pytest.mark.gpu
def test_has_lazy_gpu():
    test_has_lazy(device="cuda")
