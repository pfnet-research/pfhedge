from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Sequential

from pfhedge._utils.lazy import has_lazy


def test_has_lazy():
    m = LazyLinear(1)
    assert has_lazy(m)

    m = Linear(1, 1)
    assert not has_lazy(m)

    m = Sequential(Linear(1, 1), Linear(1, 1))
    assert not has_lazy(m)

    m = Sequential(Linear(1, 1), LazyLinear(1))
    assert has_lazy(m)
