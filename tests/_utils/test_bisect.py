import pytest
import torch
import torch.nn.functional as fn
from torch.testing import assert_close

from pfhedge._utils.bisect import bisect


def test_bisect(device: str = "cpu"):
    f = torch.sigmoid
    targets = torch.linspace(0.1, 0.9, 10).to(device)
    roots = bisect(
        f, targets, torch.tensor(-6.0).to(device), torch.tensor(6.0).to(device)
    )
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    def f(inputs):
        return -torch.sigmoid(inputs)

    targets = -torch.linspace(0.1, 0.9, 10).to(device)
    roots = bisect(
        f, targets, torch.tensor(-6.0).to(device), torch.tensor(6.0).to(device)
    )
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    f = fn.tanhshrink
    targets = torch.linspace(-0.4, 0.4, 10).to(device)
    roots = bisect(
        f,
        targets,
        torch.tensor(-6.0).to(device),
        torch.tensor(6.0).to(device),
        max_iter=100,
    )
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    def f(inputs):
        return -fn.tanhshrink(inputs)

    targets = -torch.linspace(-0.4, 0.4, 10).to(device)
    roots = bisect(
        f,
        targets,
        torch.tensor(-6.0).to(device),
        torch.tensor(6.0).to(device),
        max_iter=100,
    )
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    f = torch.tanh
    targets = torch.linspace(-0.9, 0.9, 10).to(device)
    roots = bisect(
        f,
        targets,
        torch.tensor(-6.0).to(device),
        torch.tensor(6.0).to(device),
        max_iter=100,
    )
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)

    def f(inputs):
        return -torch.tanh(inputs)

    targets = -torch.linspace(-0.9, 0.9, 10).to(device)
    roots = bisect(
        f,
        targets,
        torch.tensor(-6.0).to(device),
        torch.tensor(6.0).to(device),
        max_iter=100,
    )
    assert_close(f(roots), targets, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_bisect_gpu():
    test_bisect(device="cuda")


def test_bisect_error():
    f = torch.sigmoid
    with pytest.raises(ValueError):
        bisect(f, torch.linspace(0.1, 0.9, 10), 6.0, -6.0)

    with pytest.raises(RuntimeError):
        bisect(f, torch.linspace(0.1, 0.9, 10), -6.0, 6.0, precision=0.0, max_iter=100)
