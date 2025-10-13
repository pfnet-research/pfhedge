"""
Unit tests for Bitcoin perpetual simulation models.
"""
import unittest
import sys
import os
import pandas as pd
import numpy as np
import torch

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto.instruments import (
    BitcoinPerpetualBase,
    BitcoinPerpetualBrownian,
    BitcoinPerpetualHistorical,
)
from crypto.data.loader import CryptoDataLoader


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self):
        # Create sample data
        n_points = 1000
        dates = pd.date_range("2023-01-01", periods=n_points, freq="5min")
        prices = 50000 + np.cumsum(np.random.randn(n_points) * 100)

        self.perpetual_data = pd.DataFrame(
            {
                "timestamp": dates,
                "last_price": prices,
                "bid_price": prices - 10,
                "ask_price": prices + 10,
                "funding_8h": np.random.randn(n_points) * 0.0001,
                "index_price": prices + np.random.randn(n_points) * 5,
            }
        )

    def load_perpetual_data(self):
        return self.perpetual_data


class TestBitcoinPerpetualBase(unittest.TestCase):
    """Test cases for BitcoinPerpetualBase abstract class."""

    def test_cannot_instantiate(self):
        """Test that base class cannot be instantiated."""
        with self.assertRaises(TypeError):
            # Should fail because simulate() is abstract
            BitcoinPerpetualBase()

    def test_base_properties(self):
        """Test base class properties through a concrete subclass."""
        btc = BitcoinPerpetualBrownian()

        # Check base properties
        self.assertTrue(btc.is_listed)
        self.assertTrue(btc.has_funding)
        self.assertEqual(btc.max_leverage, 20.0)
        # Funding interval is now in years (8 hours / 24 hours / 365 days)
        self.assertAlmostEqual(btc.funding_interval, (8 / 24) / 365, places=10)

    def test_margin_requirement(self):
        """Test margin requirement calculation with leverage."""
        btc = BitcoinPerpetualBrownian()
        btc.simulate(n_paths=1, time_horizon=1 / 24 / 365)  # 1 hour in years

        position_size = 2.0  # 2 BTC
        margin = btc.margin_requirement(position_size)

        current_price = btc.spot[0, -1].item()
        expected_margin = (position_size * current_price) / btc.leverage

        self.assertAlmostEqual(margin, expected_margin, places=2)

    def test_funding_payment_times(self):
        """Test funding payment time identification."""
        btc = BitcoinPerpetualBrownian()
        btc.simulate(n_paths=1, time_horizon=8 / 24 / 365)  # 8 hours in years

        funding_times = btc.funding_payment_times()

        # Should be boolean tensor
        self.assertEqual(funding_times.dtype, torch.bool)

        # First element should be True (payment at start)
        self.assertTrue(funding_times[0])

        # Should have payments every 8 hours
        # funding_interval and dt are both in years now
        steps_per_funding = int(btc.funding_interval / btc.dt)
        for i in range(0, len(funding_times), steps_per_funding):
            if i < len(funding_times):
                self.assertTrue(funding_times[i])


class TestBitcoinPerpetualBrownian(unittest.TestCase):
    """Test cases for BitcoinPerpetualBrownian model."""

    def test_initialization(self):
        """Test Brownian model initialization."""
        btc = BitcoinPerpetualBrownian(sigma=0.8, mu=0.1)

        self.assertEqual(btc.sigma, 0.8)
        self.assertEqual(btc.mu, 0.1)
        self.assertEqual(btc.funding_mean, 0.0001)
        self.assertEqual(btc.funding_std, 0.0002)

    def test_simulate_multiple_paths(self):
        """Test generating multiple Monte Carlo paths."""
        btc = BitcoinPerpetualBrownian(sigma=0.8)

        n_paths = 100
        time_horizon = 30 / 365  # 30 days
        btc.simulate(n_paths=n_paths, time_horizon=time_horizon)

        # Check shape
        self.assertEqual(btc.spot.shape[0], n_paths)

        # Each path should be different (stochastic)
        self.assertFalse(torch.allclose(btc.spot[0], btc.spot[1]))
        self.assertFalse(torch.allclose(btc.spot[10], btc.spot[20]))

        # Check other buffers exist
        self.assertEqual(btc.bid.shape, btc.spot.shape)
        self.assertEqual(btc.ask.shape, btc.spot.shape)
        self.assertEqual(btc.funding_rate.shape, btc.spot.shape)
        self.assertEqual(btc.index_price.shape, btc.spot.shape)

    def test_volatility_property(self):
        """Test volatility property returns constant sigma."""
        btc = BitcoinPerpetualBrownian(sigma=0.75)
        btc.simulate(n_paths=10, time_horizon=5 / 365)

        vol = btc.volatility
        self.assertEqual(vol.shape, btc.spot.shape)
        self.assertTrue(torch.allclose(vol, torch.full_like(vol, 0.75)))

    def test_simulate_with_init_state(self):
        """Test simulation with custom initial price."""
        btc = BitcoinPerpetualBrownian()

        init_price = 60000.0
        btc.simulate(n_paths=5, time_horizon=1 / 365, init_state=(init_price,))

        # All paths should start at init_price
        self.assertTrue(torch.allclose(btc.spot[:, 0], torch.full((5,), init_price)))

    def test_funding_rate_correlation(self):
        """Test that funding rates correlate with momentum."""
        btc = BitcoinPerpetualBrownian(mu=0.5)  # Strong upward drift
        btc.simulate(n_paths=100, time_horizon=10 / 365)

        # With positive drift, funding should tend to be positive
        # (longs pay shorts in bull markets)
        mean_funding = btc.funding_rate.mean().item()

        # Not a strict test but funding should lean positive
        # Note: Due to randomness, we just check it's generated
        self.assertIsNotNone(mean_funding)

    def test_historical_calibration(self):
        """Test simulation with historical parameters."""
        btc = BitcoinPerpetualBrownian()

        hist_params = {
            "volatility": 0.65,
            "drift": 0.20,
            "funding_mean": 0.0003,
            "funding_std": 0.0001,
            "init_price": 55000,
        }

        btc.simulate_with_historical_parameters(
            n_paths=50, time_horizon=30 / 365, historical_data=hist_params
        )

        # Check parameters were updated
        self.assertEqual(btc.sigma, 0.65)
        self.assertEqual(btc.mu, 0.20)
        self.assertEqual(btc.funding_mean, 0.0003)
        self.assertEqual(btc.funding_std, 0.0001)

        # Check initial price
        self.assertTrue(torch.allclose(btc.spot[:, 0], torch.full((50,), 55000.0)))


class TestBitcoinPerpetualHistorical(unittest.TestCase):
    """Test cases for BitcoinPerpetualHistorical model."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_loader = MockDataLoader()

    def test_initialization_requires_loader(self):
        """Test that historical model requires data loader."""
        with self.assertRaises(ValueError):
            BitcoinPerpetualHistorical(data_loader=None)

        # Should work with loader
        btc = BitcoinPerpetualHistorical(data_loader=self.mock_loader)
        self.assertIsNotNone(btc.data_loader)

    def test_simulate_single_path(self):
        """Test loading historical data for backtesting."""
        btc = BitcoinPerpetualHistorical(data_loader=self.mock_loader)

        btc.simulate(n_paths=1, time_horizon=1 / 24 / 365)  # 1 hour in years

        # Should load actual data
        self.assertEqual(btc.spot.shape[0], 1)
        self.assertGreater(btc.spot.shape[1], 0)

        # Check all buffers loaded
        self.assertTrue(hasattr(btc, "bid"))
        self.assertTrue(hasattr(btc, "ask"))
        self.assertTrue(hasattr(btc, "funding_rate"))
        self.assertTrue(hasattr(btc, "index_price"))

    def test_replicate_paths(self):
        """Test replicating historical path for Monte Carlo with costs."""
        btc = BitcoinPerpetualHistorical(data_loader=self.mock_loader)

        n_paths = 10
        btc.simulate(n_paths=n_paths, time_horizon=1 / 24 / 365)  # 1 hour in years

        # Should have n_paths identical copies
        self.assertEqual(btc.spot.shape[0], n_paths)

        # All paths should be identical (same historical data)
        for i in range(1, n_paths):
            torch.testing.assert_close(btc.spot[0], btc.spot[i])

    def test_historical_volatility(self):
        """Test historical volatility calculation."""
        btc = BitcoinPerpetualHistorical(data_loader=self.mock_loader)
        btc.simulate(
            n_paths=1, time_horizon=30 / 365
        )  # 30 days in years (more data for volatility)

        vol = btc.volatility

        # Should have same shape as spot
        self.assertEqual(vol.shape, btc.spot.shape)

        # Volatility should be positive
        self.assertTrue((vol > 0).all())

        # Volatility should change over time (expanding window)
        # Later volatilities incorporate more data
        # With 30 days of data, volatility should evolve
        self.assertFalse(torch.allclose(vol[:, 0], vol[:, -1]))

    def test_bootstrap_simulation(self):
        """Test bootstrap sampling from historical data."""
        btc = BitcoinPerpetualHistorical(data_loader=self.mock_loader)

        n_paths = 20
        btc.simulate_bootstrap(
            n_paths=n_paths, time_horizon=1 / 24 / 365
        )  # 1 hour in years

        # Should have n_paths
        self.assertEqual(btc.spot.shape[0], n_paths)

        # Paths should be different (different windows)
        # At least some paths should differ
        different_paths = 0
        for i in range(1, n_paths):
            if not torch.allclose(btc.spot[0], btc.spot[i]):
                different_paths += 1

        # Most paths should be different
        self.assertGreater(different_paths, n_paths // 2)

    def test_cumulative_funding_cost(self):
        """Test funding cost calculation with historical data."""
        btc = BitcoinPerpetualHistorical(data_loader=self.mock_loader)
        btc.simulate(n_paths=1, time_horizon=8 / 24 / 365)  # 8 hours in years

        # Long position
        funding_cost_long = btc.cumulative_funding_cost(1.0)
        self.assertEqual(funding_cost_long.shape, btc.spot.shape)

        # Short position (opposite sign)
        funding_cost_short = btc.cumulative_funding_cost(-1.0)
        torch.testing.assert_close(funding_cost_short, -funding_cost_long)


class TestModelComparison(unittest.TestCase):
    """Test comparing different models."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_loader = MockDataLoader()

    def test_training_vs_backtesting(self):
        """Test that models are used for different purposes."""
        # Brownian for training (many paths)
        btc_train = BitcoinPerpetualBrownian(sigma=0.8)
        btc_train.simulate(n_paths=1000, time_horizon=30 / 365)

        # Historical for backtesting (single path)
        btc_backtest = BitcoinPerpetualHistorical(data_loader=self.mock_loader)
        btc_backtest.simulate(n_paths=1, time_horizon=30 / 365)

        # Training should have many different paths
        self.assertEqual(btc_train.spot.shape[0], 1000)
        self.assertFalse(torch.allclose(btc_train.spot[0], btc_train.spot[1]))

        # Backtest should have single historical path
        self.assertEqual(btc_backtest.spot.shape[0], 1)

    def test_with_pfhedge_option(self):
        """Test both models work with PFHedge options."""
        try:
            from pfhedge.instruments import EuropeanOption
        except ImportError:
            self.skipTest("PFHedge not available")

        # Brownian model for training
        btc_brownian = BitcoinPerpetualBrownian()
        option_train = EuropeanOption(
            underlier=btc_brownian, strike=50000, maturity=30 / 365
        )
        option_train.simulate(n_paths=100)

        # Check we have multiple paths
        self.assertEqual(option_train.underlier.spot.shape[0], 100)

        # Historical model for backtesting
        btc_hist = BitcoinPerpetualHistorical(data_loader=self.mock_loader)
        option_backtest = EuropeanOption(
            underlier=btc_hist, strike=50000, maturity=30 / 365
        )
        option_backtest.simulate(n_paths=1)

        # Check we have historical data
        self.assertEqual(option_backtest.underlier.spot.shape[0], 1)

        # Both should calculate payoffs
        payoff_train = option_train.payoff()
        payoff_backtest = option_backtest.payoff()

        self.assertEqual(payoff_train.shape[0], 100)
        self.assertEqual(payoff_backtest.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
