from math import sqrt

import pytest
import torch
from torch.testing import assert_close

from pfhedge.stochastic import generate_brownian
from pfhedge.stochastic import generate_geometric_brownian
from pfhedge.stochastic.engine import RandnSobolBoxMuller


def test_generate_brownian_mean(device: str = "cpu"):
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250

    device = torch.device(device) if device else None
    output = generate_brownian(n_paths, n_steps, device=device)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.zeros_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


@pytest.mark.gpu
def test_generate_brownian_mean_gpu():
    test_generate_brownian_mean(device="cuda")


def test_generate_brownian_nosigma(device: str = "cpu"):
    torch.manual_seed(42)
    n_steps = 250

    device = torch.device(device) if device else None
    result = generate_brownian(1, n_steps, sigma=0, device=device)
    expect = torch.zeros(1, n_steps).to(device)
    assert_close(result, expect)

    mu = 0.1
    dt = 0.01
    result = generate_brownian(1, n_steps, mu=mu, sigma=0, dt=dt, device=device)
    expect = torch.linspace(0, mu * dt * (n_steps - 1), n_steps).to(device).unsqueeze(0)
    assert_close(result, expect)


@pytest.mark.gpu
def test_generate_brownian_nosigma_gpu():
    test_generate_brownian_nosigma(device="cuda")


def test_generate_brownian_std(device: str = "cpu"):
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250

    device = torch.device(device) if device else None
    output = generate_brownian(n_paths, n_steps, device=device)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].std()
    expect = torch.full_like(result, 0.2)
    assert_close(result, expect, atol=0, rtol=0.1)


@pytest.mark.gpu
def test_generate_brownian_std_gpu():
    test_generate_brownian_std(device="cuda")


def test_generate_brownian_mean_init_state(device: str = "cpu"):
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250

    device = torch.device(device) if device else None
    output = generate_brownian(n_paths, n_steps, init_state=1.0, device=device)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)

    output = generate_brownian(
        n_paths, n_steps, init_state=torch.tensor(1.0).to(device), device=device
    )
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)

    output = generate_brownian(
        n_paths, n_steps, init_state=torch.tensor([1.0]).to(device), device=device
    )
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.ones_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


@pytest.mark.gpu
def test_generate_brownian_mean_init_state_gpu():
    test_generate_brownian_mean_init_state(device="cuda")


def test_generate_brownian_mean_mu(device: str = "cpu"):
    torch.manual_seed(42)
    n_paths = 10000
    n_steps = 250
    dt = 1 / 250
    mu = 0.1

    device = torch.device(device) if device else None
    output = generate_brownian(n_paths, n_steps, mu=mu, device=device)
    result = output[:, -1].mean()
    expect = torch.full_like(result, mu * dt * n_steps)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


@pytest.mark.gpu
def test_generate_brownian_mean_mu_gpu():
    test_generate_brownian_mean_mu(device="cuda")


def test_generate_brownian_sobol_mean(device: str = "cpu"):
    n_paths = 10000
    n_steps = 250

    device = torch.device(device) if device else None
    engine = RandnSobolBoxMuller(seed=42, scramble=True)
    output = generate_brownian(n_paths, n_steps, engine=engine, device=device)
    assert output.size() == torch.Size((n_paths, n_steps))
    result = output[:, -1].mean()
    expect = torch.zeros_like(result)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


@pytest.mark.gpu
def test_generate_brownian_sobol_mean_gpu():
    test_generate_brownian_sobol_mean(device="cuda")


def test_generate_brownian_dtype(device: str = "cpu"):
    torch.manual_seed(42)

    device = torch.device(device) if device else None
    output = generate_brownian(1, 1, dtype=torch.float32, device=device)
    assert output.dtype == torch.float32

    output = generate_brownian(1, 1, dtype=torch.float64, device=device)
    assert output.dtype == torch.float64


@pytest.mark.gpu
def test_generate_brownian_dtype_gpu():
    test_generate_brownian_dtype(device="cuda")


def test_generate_geometric_brownian_mean(device: str = "cpu"):
    torch.manual_seed(42)

    n_paths = 10000
    n_steps = 250

    device = torch.device(device) if device else None
    t = generate_geometric_brownian(n_paths, n_steps, device=device)
    result = t[:, -1].mean()
    expect = torch.ones_like(result).to(device)
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)

    t = generate_geometric_brownian(n_paths, n_steps, init_state=2.0, device=device)
    result = t[:, -1].mean()
    expect = torch.ones_like(result).to(device) * 2
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)

    mu = 0.1
    dt = 1 / 250

    output = generate_geometric_brownian(n_paths, n_steps, mu=mu, device=device)
    result = output[:, -1].mean()
    expect = torch.ones_like(result) * torch.tensor(n_steps * dt * mu).to(device).exp()
    std = 0.2 * sqrt(1 / n_paths)
    assert_close(result, expect, atol=3 * std, rtol=0)


@pytest.mark.gpu
def test_generate_geometric_brownian_mean_gpu():
    test_generate_geometric_brownian_mean(device="cuda")


def test_generate_geometric_brownian_dtype(device: str = "cpu"):
    torch.manual_seed(42)

    device = torch.device(device) if device else None
    output = generate_geometric_brownian(1, 1, dtype=torch.float32, device=device)
    assert output.dtype == torch.float32

    output = generate_geometric_brownian(1, 1, dtype=torch.float64, device=device)
    assert output.dtype == torch.float64


@pytest.mark.gpu
def test_generate_geometric_brownian_dtype_gpu():
    test_generate_geometric_brownian_dtype(device="cuda")
