"""Unit tests for GBM simulator sanity checks.

This module verifies the Geometric Brownian Motion simulator used for deep hedging
satisfies fundamental mathematical properties:
1. Martingale property under risk-neutral measure
2. Correct drift and volatility
3. Lognormal distribution of returns
4. Path independence
"""

import pytest
import torch
import numpy as np

from crypto.instruments import BitcoinPerpetualBrownian


class TestMartingaleProperty:
    """Test martingale property of GBM under risk-neutral measure."""

    def test_zero_drift_martingale(self):
        """Test that with zero drift, expected future spot equals initial spot."""
        torch.manual_seed(42)

        # Create GBM with zero drift (mu=0) - risk-neutral measure
        underlier = BitcoinPerpetualBrownian(
            dt=1 / 252,  # Daily time step
            sigma=0.2,  # 20% annual volatility
            mu=0.0,  # Zero drift for martingale property
        )

        # Simulate many paths
        n_paths = 10000
        time_horizon = 1.0  # 1 year
        init_spot = 1.0

        underlier.simulate(
            n_paths=n_paths, time_horizon=time_horizon, init_state=(init_spot,)
        )

        # Check martingale property: E[S_T] = S_0 under risk-neutral measure
        final_spots = underlier.spot[:, -1]
        expected_final = final_spots.mean().item()

        # With large sample, expected value should be close to initial
        # Allow 5% error margin (more paths would reduce this)
        relative_error = abs(expected_final - init_spot) / init_spot
        assert (
            relative_error < 0.05
        ), f"Martingale property violated: E[S_T]={expected_final:.4f} vs S_0={init_spot:.4f} (error: {relative_error:.2%})"

    def test_intermediate_martingale(self):
        """Test martingale property at intermediate time steps."""
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=1 / 252, sigma=0.2, mu=0.0)

        n_paths = 5000
        underlier.simulate(n_paths=n_paths, time_horizon=0.5, init_state=(1.0,))

        # Check E[S_t] ≈ S_0 for all t
        spots = underlier.spot
        init_spot = spots[:, 0].mean().item()

        # Check at quarter points
        n_steps = spots.shape[1]
        check_points = [n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]

        for step in check_points:
            expected_spot = spots[:, step].mean().item()
            relative_error = abs(expected_spot - init_spot) / init_spot

            assert (
                relative_error < 0.1
            ), f"Martingale violated at step {step}: E[S_t]={expected_spot:.4f} vs S_0={init_spot:.4f}"


class TestVolatilityCalibration:
    """Test that simulated volatility matches specified volatility."""

    def test_realized_vol_matches_sigma(self):
        """Test that realized volatility converges to specified sigma."""
        torch.manual_seed(42)

        sigma = 0.8  # 80% annual volatility (crypto-like)
        underlier = BitcoinPerpetualBrownian(dt=8 / 24 / 365, sigma=sigma, mu=0.0)

        # Simulate long paths with many samples
        n_paths = 5000
        time_horizon = 1.0  # 1 year
        underlier.simulate(
            n_paths=n_paths, time_horizon=time_horizon, init_state=(1.0,)
        )

        # Calculate realized volatility from log returns
        spots = underlier.spot
        log_returns = torch.log(spots[:, 1:] / spots[:, :-1])

        # Annualize: realized_vol = std(log_returns) * sqrt(1/dt)
        dt = underlier.dt
        annualization_factor = np.sqrt(1.0 / dt)
        realized_vol = log_returns.std().item() * annualization_factor

        # Check realized vol is close to specified sigma
        # Allow 10% error margin (more paths/longer horizon would reduce this)
        relative_error = abs(realized_vol - sigma) / sigma
        assert (
            relative_error < 0.1
        ), f"Volatility mismatch: realized={realized_vol:.4f} vs specified={sigma:.4f} (error: {relative_error:.2%})"

    def test_vol_scales_with_time_step(self):
        """Test that volatility properly scales with different time steps."""
        torch.manual_seed(42)

        sigma = 0.5
        n_paths = 3000

        # Test with different time steps
        dt_values = [1 / 252, 1 / 52, 1 / 12]  # Daily, weekly, monthly
        realized_vols = []

        for dt in dt_values:
            underlier = BitcoinPerpetualBrownian(dt=dt, sigma=sigma, mu=0.0)
            underlier.simulate(n_paths=n_paths, time_horizon=1.0, init_state=(1.0,))

            spots = underlier.spot
            log_returns = torch.log(spots[:, 1:] / spots[:, :-1])
            annualization_factor = np.sqrt(1.0 / dt)
            realized_vol = log_returns.std().item() * annualization_factor

            realized_vols.append(realized_vol)

        # All realized vols should be close to sigma
        for i, (dt, realized_vol) in enumerate(zip(dt_values, realized_vols)):
            relative_error = abs(realized_vol - sigma) / sigma
            assert (
                relative_error < 0.15
            ), f"Volatility not scaling correctly for dt={dt}: realized={realized_vol:.4f} vs sigma={sigma:.4f}"


class TestLognormalDistribution:
    """Test that GBM produces lognormally distributed prices."""

    def test_log_returns_normality(self):
        """Test that log returns are approximately normally distributed."""
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=1 / 252, sigma=0.3, mu=0.0)

        # Simulate many paths
        n_paths = 10000
        underlier.simulate(n_paths=n_paths, time_horizon=0.5, init_state=(1.0,))

        # Calculate all log returns
        spots = underlier.spot
        log_returns = torch.log(spots[:, 1:] / spots[:, :-1])

        # Flatten to get all returns
        all_returns = log_returns.flatten()

        # Test for normality: check skewness and kurtosis
        mean = all_returns.mean().item()
        std = all_returns.std().item()

        # For normal distribution:
        # - Skewness should be ~0
        # - Excess kurtosis should be ~0 (total kurtosis ~3)

        # Calculate standardized moments
        centered = (all_returns - mean) / std
        skewness = (centered**3).mean().item()
        kurtosis = (centered**4).mean().item()

        # Allow generous bounds for large sample
        assert abs(skewness) < 0.2, f"Log returns not normal: skewness={skewness:.4f}"
        assert (
            abs(kurtosis - 3.0) < 1.0
        ), f"Log returns not normal: kurtosis={kurtosis:.4f}"

    def test_final_price_lognormal(self):
        """Test that final prices follow lognormal distribution."""
        torch.manual_seed(42)

        sigma = 0.4
        mu = 0.0
        T = 1.0
        S0 = 1.0

        underlier = BitcoinPerpetualBrownian(dt=1 / 252, sigma=sigma, mu=mu)
        n_paths = 10000
        underlier.simulate(n_paths=n_paths, time_horizon=T, init_state=(S0,))

        final_prices = underlier.spot[:, -1]

        # For lognormal, log(S_T) should be normal with:
        # mean = log(S_0) + (mu - 0.5*sigma^2)*T
        # std = sigma * sqrt(T)
        log_prices = torch.log(final_prices)

        expected_mean = np.log(S0) + (mu - 0.5 * sigma**2) * T
        expected_std = sigma * np.sqrt(T)

        actual_mean = log_prices.mean().item()
        actual_std = log_prices.std().item()

        # Check mean
        mean_error = abs(actual_mean - expected_mean) / abs(expected_mean + 1e-8)
        assert (
            mean_error < 0.1
        ), f"Log price mean mismatch: actual={actual_mean:.4f} vs expected={expected_mean:.4f}"

        # Check std
        std_error = abs(actual_std - expected_std) / expected_std
        assert (
            std_error < 0.1
        ), f"Log price std mismatch: actual={actual_std:.4f} vs expected={expected_std:.4f}"


class TestPathIndependence:
    """Test that simulated paths are independent."""

    def test_different_seeds_different_paths(self):
        """Test that different seeds produce different paths."""
        sigma = 0.6

        # Seed 1
        torch.manual_seed(42)
        underlier1 = BitcoinPerpetualBrownian(dt=1 / 252, sigma=sigma, mu=0.0)
        underlier1.simulate(n_paths=100, time_horizon=0.25, init_state=(1.0,))
        spots1 = underlier1.spot.clone()

        # Seed 2
        torch.manual_seed(123)
        underlier2 = BitcoinPerpetualBrownian(dt=1 / 252, sigma=sigma, mu=0.0)
        underlier2.simulate(n_paths=100, time_horizon=0.25, init_state=(1.0,))
        spots2 = underlier2.spot.clone()

        # Paths should be different
        assert not torch.allclose(
            spots1, spots2
        ), "Different seeds should produce different paths"

    def test_same_seed_same_paths(self):
        """Test that same seed produces identical paths."""
        sigma = 0.6

        # Run 1
        torch.manual_seed(42)
        underlier1 = BitcoinPerpetualBrownian(dt=1 / 252, sigma=sigma, mu=0.0)
        underlier1.simulate(n_paths=100, time_horizon=0.25, init_state=(1.0,))
        spots1 = underlier1.spot.clone()

        # Run 2 with same seed
        torch.manual_seed(42)
        underlier2 = BitcoinPerpetualBrownian(dt=1 / 252, sigma=sigma, mu=0.0)
        underlier2.simulate(n_paths=100, time_horizon=0.25, init_state=(1.0,))
        spots2 = underlier2.spot.clone()

        # Paths should be identical
        assert torch.allclose(
            spots1, spots2
        ), "Same seed should produce identical paths"

    def test_path_correlation_near_zero(self):
        """Test that different paths have near-zero correlation."""
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=1 / 252, sigma=0.4, mu=0.0)
        underlier.simulate(n_paths=100, time_horizon=0.5, init_state=(1.0,))

        spots = underlier.spot

        # Calculate correlation between first two paths
        path1 = spots[0, :]
        path2 = spots[1, :]

        # Pearson correlation
        corr = torch.corrcoef(torch.stack([path1, path2]))[0, 1].item()

        # Correlation should be close to zero (allow some random variation)
        assert abs(corr) < 0.3, f"Paths should be independent: correlation={corr:.4f}"


class TestNumericalStability:
    """Test numerical stability of GBM simulation."""

    def test_no_nan_values(self):
        """Test that simulation produces no NaN values."""
        torch.manual_seed(42)

        # Test with various parameter combinations
        test_cases = [
            {"sigma": 0.2, "mu": 0.0, "dt": 1 / 252},
            {"sigma": 1.0, "mu": 0.05, "dt": 1 / 52},
            {"sigma": 0.5, "mu": -0.02, "dt": 1 / 12},
        ]

        for params in test_cases:
            underlier = BitcoinPerpetualBrownian(**params)
            underlier.simulate(n_paths=100, time_horizon=1.0, init_state=(1.0,))

            assert not torch.any(
                torch.isnan(underlier.spot)
            ), f"NaN values found with params: {params}"

    def test_no_inf_values(self):
        """Test that simulation produces no infinite values."""
        torch.manual_seed(42)

        # Test with high volatility
        underlier = BitcoinPerpetualBrownian(dt=1 / 252, sigma=2.0, mu=0.0)
        underlier.simulate(n_paths=100, time_horizon=1.0, init_state=(1.0,))

        assert not torch.any(
            torch.isinf(underlier.spot)
        ), "Infinite values found in simulation"

    def test_positive_prices(self):
        """Test that all simulated prices remain positive."""
        torch.manual_seed(42)

        underlier = BitcoinPerpetualBrownian(dt=1 / 252, sigma=0.8, mu=0.0)
        underlier.simulate(n_paths=1000, time_horizon=1.0, init_state=(1.0,))

        assert torch.all(
            underlier.spot > 0
        ), "GBM should produce strictly positive prices"


def test_gbm_sanity_summary():
    """Summary test for GBM simulator sanity checks."""
    torch.manual_seed(42)

    # Create simulator
    sigma = 0.6
    mu = 0.0
    underlier = BitcoinPerpetualBrownian(dt=1 / 252, sigma=sigma, mu=mu)

    # Simulate
    n_paths = 5000
    time_horizon = 1.0
    init_spot = 1.0
    underlier.simulate(
        n_paths=n_paths, time_horizon=time_horizon, init_state=(init_spot,)
    )

    # Test 1: Martingale property
    final_spots = underlier.spot[:, -1]
    expected_final = final_spots.mean().item()
    martingale_error = abs(expected_final - init_spot) / init_spot
    assert (
        martingale_error < 0.05
    ), f"Martingale property violated: error={martingale_error:.2%}"

    # Test 2: Volatility calibration
    log_returns = torch.log(underlier.spot[:, 1:] / underlier.spot[:, :-1])
    dt = underlier.dt
    realized_vol = log_returns.std().item() * np.sqrt(1.0 / dt)
    vol_error = abs(realized_vol - sigma) / sigma
    assert vol_error < 0.1, f"Volatility mismatch: error={vol_error:.2%}"

    # Test 3: No NaN/Inf
    assert not torch.any(torch.isnan(underlier.spot)), "NaN values detected"
    assert not torch.any(torch.isinf(underlier.spot)), "Inf values detected"

    # Test 4: Positive prices
    assert torch.all(underlier.spot > 0), "Non-positive prices detected"

    print("\n✅ GBM Simulator Sanity Checks PASSED!")
    print(f"   Martingale error: {martingale_error:.2%}")
    print(f"   Volatility error: {vol_error:.2%}")
    print(f"   Realized vol: {realized_vol:.4f} vs specified: {sigma:.4f}")
    print(f"   E[S_T]: {expected_final:.4f} vs S_0: {init_spot:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
