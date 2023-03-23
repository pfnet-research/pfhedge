from math import sqrt

import pytest
import torch
from torch.distributions.gamma import Gamma
from torch.testing import assert_close

from pfhedge.stochastic import generate_cir


def test_generate_cir_mean_1(device: str = "cpu"):
    torch.manual_seed(42)

    n_paths = 10000
    theta = 0.04
    sigma = 2.0
    kappa = 1.0

    device = torch.device(device) if device else None
    t = generate_cir(n_paths, 250, kappa=kappa, theta=theta, sigma=sigma, device=device)
    result = t[:, -1].mean()
    # Asymptotic distribution is gamma distribution
    alpha = 2 * kappa * theta / sigma ** 2
    beta = 2 * kappa / sigma ** 2
    d = Gamma(alpha, beta)

    expect = torch.full_like(result, d.mean)
    std = sqrt(d.variance / n_paths)

    assert_close(result, expect, atol=3 * std, rtol=0)


@pytest.mark.gpu
def test_generate_cir_mean_1_gpu():
    test_generate_cir_mean_1(device="cuda")


def test_generate_cir_mean_2(device: str = "cpu"):
    torch.manual_seed(42)

    n_paths = 10000
    theta = 0.04
    sigma = 2.0
    kappa = 1.0

    device = torch.device(device) if device else None
    t = generate_cir(
        n_paths,
        250,
        init_state=0.05,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        device=device,
    )
    result = t[:, -1].mean()
    # Asymptotic distribution is gamma distribution
    alpha = 2 * kappa * theta / sigma ** 2
    beta = 2 * kappa / sigma ** 2
    d = Gamma(alpha, beta)

    expect = torch.full_like(result, d.mean)
    std = sqrt(d.variance / n_paths)

    assert_close(result, expect, atol=3 * std, rtol=0)


@pytest.mark.gpu
def test_generate_cir_mean_2_gpu():
    test_generate_cir_mean_2(device="cuda")


def test_dtype(device: str = "cpu"):
    device = torch.device(device) if device else None

    output = generate_cir(2, 3, dtype=torch.float32, device=device)
    assert output.dtype == torch.float32

    output = generate_cir(2, 3, dtype=torch.float64, device=device)
    assert output.dtype == torch.float64


@pytest.mark.gpu
def test_dtype_gpu():
    test_dtype(device="cuda")
