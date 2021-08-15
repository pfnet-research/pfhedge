from math import sqrt

import torch
from torch.testing import assert_close

from pfhedge.stochastic import generate_brownian
from pfhedge.stochastic import generate_geometric_brownian


def test_generate_brownian_mean():
    torch.manual_seed(42)

    n_paths = 10000
    t = generate_brownian(n_paths, 250)
    result = t[:, -1].mean()
    expect = torch.zeros_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)

    torch.manual_seed(42)

    n_paths = 10000
    t = generate_brownian(n_paths, 250)
    result = t[:, -1].std()
    expect = torch.full_like(result, 0.2)
    assert_close(result, expect, atol=0, rtol=0.1)

    torch.manual_seed(42)

    n_paths = 10000
    t = generate_brownian(n_paths, 250, init_state=1.0)
    result = t[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


def test_generate_brownian_dtype():
    output = generate_brownian(1, 1, dtype=torch.float32)
    assert output.dtype == torch.float32

    output = generate_brownian(1, 1, dtype=torch.float64)
    assert output.dtype == torch.float64


def test_generate_geometric_brownian_mean():
    torch.manual_seed(42)

    n_paths = 10000
    t = generate_geometric_brownian(n_paths, 250)
    result = t[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)

    torch.manual_seed(42)

    n_paths = 10000
    t = generate_geometric_brownian(n_paths, 250, init_state=2.0)
    result = t[:, -1].mean()
    expect = torch.ones_like(result) * 2
    std = 0.2 * sqrt(1 / n_paths)
    print(result)
    print(expect)
    assert_close(result, expect, atol=3 * std, rtol=0)


def test_generate_geometric_brownian_dtype():
    output = generate_geometric_brownian(1, 1, dtype=torch.float32)
    assert output.dtype == torch.float32

    output = generate_geometric_brownian(1, 1, dtype=torch.float64)
    assert output.dtype == torch.float64
