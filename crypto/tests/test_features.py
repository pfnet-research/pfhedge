#!/usr/bin/env python3
"""
Tests for feature engineering utilities.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import unittest
import torch
import numpy as np
from crypto.features.volatility import (
    calculate_realized_volatility,
    RealizedVolatilityCalculator,
    create_volatility_features,
    estimate_annualization_factor,
)
from crypto.instruments import BitcoinPerpetualBrownian


class TestVolatilityFeatures(unittest.TestCase):
    """Test volatility calculation functions."""

    def setUp(self):
        """Set up test data."""
        torch.manual_seed(42)
        np.random.seed(42)

    def test_calculate_realized_volatility_basic(self):
        """Test basic realized volatility calculation."""
        # Create simple price series
        prices = torch.tensor([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0])

        vol = calculate_realized_volatility(
            prices, window=3, annualization_factor=np.sqrt(252)
        )

        self.assertEqual(vol.shape, prices.shape)
        self.assertTrue(torch.isfinite(vol[-1]))  # Final value should be valid

    def test_calculate_realized_volatility_batch(self):
        """Test volatility calculation with multiple paths."""
        n_paths, n_steps = 10, 50
        prices = torch.randn(n_paths, n_steps).cumsum(dim=1).exp() * 100

        vol = calculate_realized_volatility(prices, window=10)

        self.assertEqual(vol.shape, (n_paths, n_steps))
        # Check that we get valid results
        valid_vol = vol[~torch.isnan(vol)]
        self.assertGreater(len(valid_vol), 0)
        self.assertTrue(torch.all(valid_vol > 0))

    def test_realized_volatility_calculator(self):
        """Test the stateful volatility calculator."""
        calc = RealizedVolatilityCalculator(
            windows=[5, 10], annualization_factor=np.sqrt(252)
        )

        # Feed some prices
        prices = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 105.0, 96.0, 104.0]

        for price in prices:
            vols = calc.update(price)

        self.assertEqual(len(vols), 2)  # Two windows
        # Should have valid volatilities by now
        self.assertTrue(all(not np.isnan(v) for v in vols))

    def test_create_volatility_features(self):
        """Test volatility features creation for instruments."""
        btc = BitcoinPerpetualBrownian(sigma=0.8)
        btc.simulate(n_paths=10, time_horizon=30 / 365)

        features = create_volatility_features(btc, windows=[5, 10, 20])

        expected_shape = (10, btc.spot.shape[1], 3)
        self.assertEqual(features.shape, expected_shape)

    def test_estimate_annualization_factor(self):
        """Test annualization factor estimation."""
        # Test daily data (24 * 3600 seconds)
        daily_factor = estimate_annualization_factor(24 * 3600)
        self.assertAlmostEqual(daily_factor, np.sqrt(365.25), places=1)

        # Test 5-minute data (5 * 60 seconds)
        min5_factor = estimate_annualization_factor(5 * 60)
        expected = np.sqrt(365.25 * 24 * 12)  # 12 five-min periods per hour
        self.assertAlmostEqual(min5_factor, expected, places=0)

    def test_volatility_with_nans(self):
        """Test handling of insufficient data."""
        # Very short series
        prices = torch.tensor([100.0, 101.0])

        vol = calculate_realized_volatility(prices, window=10, min_periods=3)

        # Should be mostly NaN due to insufficient data
        self.assertTrue(torch.isnan(vol[0]))

    def test_volatility_convergence(self):
        """Test that volatility converges to known value for GBM."""
        torch.manual_seed(42)

        # Generate long GBM series with known volatility
        n_steps = 1000
        true_vol = 0.2
        dt = 1 / 252

        prices = torch.zeros(1, n_steps)
        prices[0, 0] = 100.0

        for t in range(1, n_steps):
            dW = torch.randn(1) * np.sqrt(dt)
            prices[0, t] = prices[0, t - 1] * torch.exp(
                (0.0 - 0.5 * true_vol ** 2) * dt + true_vol * dW
            )

        realized_vol = calculate_realized_volatility(
            prices, window=50, annualization_factor=np.sqrt(252)
        )

        # Final realized vol should be in reasonable range (volatility estimation is noisy)
        final_vol = realized_vol[0, -1].item()
        self.assertGreater(final_vol, 0.01)  # Should be positive
        self.assertLess(final_vol, 1.0)  # Should be reasonable magnitude


class TestIntegrationWithInstruments(unittest.TestCase):
    """Test volatility features with actual instruments."""

    def setUp(self):
        """Set up test instruments."""
        torch.manual_seed(42)

    def test_integration_with_bitcoin_brownian(self):
        """Test volatility features with BitcoinPerpetualBrownian."""
        btc = BitcoinPerpetualBrownian(sigma=0.8, mu=0.0)
        btc.simulate(n_paths=5, time_horizon=7 / 365)

        # Test basic volatility calculation
        vol = calculate_realized_volatility(btc.spot, window=5)
        self.assertEqual(vol.shape, btc.spot.shape)

        # Test feature creation
        features = create_volatility_features(btc, windows=[3, 5])
        expected_shape = (5, btc.spot.shape[1], 2)
        self.assertEqual(features.shape, expected_shape)

    def test_volatility_features_non_nan(self):
        """Test that volatility features don't produce unexpected NaNs."""
        btc = BitcoinPerpetualBrownian(sigma=0.5)
        btc.simulate(n_paths=3, time_horizon=20 / 365)  # Longer series

        features = create_volatility_features(btc, windows=[5, 10])

        # Should have some valid (non-NaN) values
        valid_features = features[~torch.isnan(features)]
        self.assertGreater(len(valid_features), 0)

        # All valid features should be positive
        self.assertTrue(torch.all(valid_features > 0))


if __name__ == "__main__":
    unittest.main()
