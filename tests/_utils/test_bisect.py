import pytest
import torch
import torch.nn.functional as fn
from torch.testing import assert_close

from pfhedge._utils.bisect import bisect


def test_bisect():
    f = torch.sigmoid
    targets = torch.linspace(0.1, 0.9, 10)
    roots = bisect(f, targets, -6.0, 6.0)
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    f = lambda input: -torch.sigmoid(input)
    targets = -torch.linspace(0.1, 0.9, 10)
    roots = bisect(f, targets, -6.0, 6.0)
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    f = fn.tanhshrink
    targets = torch.linspace(-0.4, 0.4, 10)
    roots = bisect(f, targets, -6.0, 6.0, abort=100)
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    f = lambda input: -fn.tanhshrink(input)
    targets = -torch.linspace(-0.4, 0.4, 10)
    roots = bisect(f, targets, -6.0, 6.0, abort=100)
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    f = torch.tanh
    targets = torch.linspace(-0.9, 0.9, 10)
    roots = bisect(f, targets, -6.0, 6.0, abort=100)
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    f = lambda input: -torch.tanh(input)
    targets = -torch.linspace(-0.9, 0.9, 10)
    roots = bisect(f, targets, -6.0, 6.0, abort=100)
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)


def test_bisect_error():
    f = torch.sigmoid
    with pytest.raises(ValueError):
        bisect(f, torch.linspace(0.1, 0.9, 10), 6.0, -6.0)

    with pytest.raises(RuntimeError):
        bisect(f, torch.linspace(0.1, 0.9, 10), -6.0, 6.0, precision=0.0, abort=100)
