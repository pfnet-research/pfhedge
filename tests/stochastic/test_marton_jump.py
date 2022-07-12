from math import sqrt

import torch
from torch.testing import assert_close

from pfhedge.stochastic import generate_marton_jump
from pfhedge.stochastic.engine import RandnSobolBoxMuller


def test_generate_brownian_mean_no_jump():
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250

    output = generate_marton_jump(n_paths, n_steps, jump_std=0.0)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


def test_generate_brownian_mean_no_jump1():
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250

    output = generate_marton_jump(n_paths, n_steps, jump_per_year=0.0)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


def test_generate_brownian_mean():
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250

    output = generate_marton_jump(n_paths, n_steps, jump_per_year=100)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].log().mean()
    expect = torch.zeros_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


def test_generate_marton_jump_nosigma():
    torch.manual_seed(42)
    n_steps = 250

    result = generate_marton_jump(1, n_steps, sigma=0, jump_per_year=0)
    expect = torch.ones(1, n_steps)
    assert_close(result, expect)

    mu = 0.1
    dt = 0.01
    result = generate_marton_jump(
        1, n_steps, mu=mu, sigma=0, dt=dt, jump_per_year=0
    ).log()
    expect = torch.linspace(0, mu * dt * (n_steps - 1), n_steps).unsqueeze(0)
    assert_close(result, expect)


def test_generate_marton_jump_nosigma2():
    torch.manual_seed(42)
    n_steps = 250

    result = generate_marton_jump(1, n_steps, sigma=0, jump_std=0)
    expect = torch.ones(1, n_steps)
    assert_close(result, expect)

    mu = 0.1
    dt = 0.01
    result = generate_marton_jump(1, n_steps, mu=mu, sigma=0, dt=dt, jump_std=0).log()
    expect = torch.linspace(0, mu * dt * (n_steps - 1), n_steps).unsqueeze(0)
    assert_close(result, expect)


def test_generate_marton_jump_std():
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250

    output = generate_marton_jump(n_paths, n_steps, jump_per_year=0)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].log().std()
    expect = torch.full_like(result, 0.2)
    assert_close(result, expect, atol=0, rtol=0.1)


def test_generate_marton_jump_std2():
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250

    output = generate_marton_jump(n_paths, n_steps, jump_std=0)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].log().std()
    expect = torch.full_like(result, 0.2)
    assert_close(result, expect, atol=0, rtol=0.1)


def test_generate_marton_jump_mean_init_state():
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250

    output = generate_marton_jump(n_paths, n_steps, init_state=1.0, jump_per_year=0)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)

    output = generate_marton_jump(
        n_paths, n_steps, init_state=torch.tensor(1.0), jump_per_year=0
    )
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)

    output = generate_marton_jump(
        n_paths, n_steps, init_state=torch.tensor([1.0]), jump_per_year=0
    )
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


def test_generate_marton_jump_mean_mu():
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250
    dt = 1 / 250
    mu = 0.1

    output = generate_marton_jump(n_paths, n_steps, mu=mu, jump_per_year=0)
    result = output[:, -1].mean().log()
    expect = torch.full_like(result, mu * dt * n_steps)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


def test_generate_marton_jump_dtype():
    torch.manual_seed(42)

    output = generate_marton_jump(1, 1, dtype=torch.float32)
    assert output.dtype == torch.float32

    output = generate_marton_jump(1, 1, dtype=torch.float64)
    assert output.dtype == torch.float64
