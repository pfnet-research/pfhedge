from math import sqrt

import torch
from torch.distributions.gamma import Gamma
from torch.testing import assert_close

from pfhedge.stochastic import generate_cir


def test_generate_cir_mean_1():
    torch.manual_seed(42)

    n_paths = 10000
    theta = 0.04
    sigma = 2.0
    kappa = 1.0

    t = generate_cir(n_paths, 250, kappa=kappa, theta=theta, sigma=sigma)
    result = t[:, -1].mean()
    # Asymptotic distribution is gamma distribution
    alpha = 2 * kappa * theta / sigma ** 2
    beta = 2 * kappa / sigma ** 2
    d = Gamma(alpha, beta)

    expect = torch.full_like(result, d.mean)
    std = sqrt(d.variance / n_paths)

    assert_close(result, expect, atol=3 * std, rtol=0)


def test_generate_cir_mean_2():
    torch.manual_seed(42)

    n_paths = 10000
    theta = 0.04
    sigma = 2.0
    kappa = 1.0

    t = generate_cir(
        n_paths, 250, init_state=0.05, kappa=kappa, theta=theta, sigma=sigma
    )
    result = t[:, -1].mean()
    # Asymptotic distribution is gamma distribution
    alpha = 2 * kappa * theta / sigma ** 2
    beta = 2 * kappa / sigma ** 2
    d = Gamma(alpha, beta)

    expect = torch.full_like(result, d.mean)
    std = sqrt(d.variance / n_paths)

    assert_close(result, expect, atol=3 * std, rtol=0)


def test_dtype():
    output = generate_cir(2, 3, dtype=torch.float32)
    assert output.dtype == torch.float32

    output = generate_cir(2, 3, dtype=torch.float64)
    assert output.dtype == torch.float64
