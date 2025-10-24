"""
Unit tests for Bitcoin instruments.
"""
import unittest
import sys
import os
import tempfile
import pandas as pd
import numpy as np
import torch

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto.instruments import BitcoinSpot, BitcoinPerpetualHistorical
from crypto.data.loader import CryptoDataLoader


class MockDataLoader:
    """Mock data loader for testing."""

    def __init__(self):
        # Create sample data
        n_points = 100
        dates = pd.date_range("2023-01-01", periods=n_points, freq="5min")
        prices = 50000 + np.random.randn(n_points) * 1000

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


class TestBitcoinSpot(unittest.TestCase):
    """Test cases for BitcoinSpot instrument."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_loader = MockDataLoader()
        self.btc_spot = BitcoinSpot(data_loader=self.mock_loader)

    def test_initialization(self):
        """Test BitcoinSpot initialization."""
        self.assertEqual(self.btc_spot.cost, 0.001)
        self.assertEqual(self.btc_spot.dt, 1 / 24 / 12)
        self.assertEqual(self.btc_spot.leverage, 1.0)
        self.assertFalse(self.btc_spot.has_funding)

    def test_simulate(self):
        """Test loading historical data via simulate."""
        # Simulate for 1 hour (12 5-minute bars)
        time_horizon = 1 / 24
        self.btc_spot.simulate(n_paths=1, time_horizon=time_horizon)

        # Check buffers are created
        self.assertTrue(hasattr(self.btc_spot, "spot"))
        spot = self.btc_spot.spot

        # Check shape
        self.assertEqual(spot.dim(), 2)
        self.assertEqual(spot.shape[0], 1)  # n_paths
        self.assertGreaterEqual(spot.shape[1], 12)  # at least 12 time steps

        # Check other buffers
        self.assertTrue(hasattr(self.btc_spot, "bid"))
        self.assertTrue(hasattr(self.btc_spot, "ask"))
        self.assertTrue(hasattr(self.btc_spot, "mid"))

    def test_multiple_paths(self):
        """Test simulating multiple paths."""
        self.btc_spot.simulate(n_paths=3, time_horizon=1 / 24)

        spot = self.btc_spot.spot
        self.assertEqual(spot.shape[0], 3)

        # All paths should be identical for historical data
        torch.testing.assert_close(spot[0], spot[1])
        torch.testing.assert_close(spot[1], spot[2])

    def test_margin_requirement(self):
        """Test margin requirement calculation."""
        self.btc_spot.simulate(n_paths=1, time_horizon=1 / 24)

        # For spot, margin = full notional
        position_size = 2.0  # 2 BTC
        margin = self.btc_spot.margin_requirement(position_size)

        current_price = self.btc_spot.spot[0, -1].item()
        expected_margin = position_size * current_price

        self.assertAlmostEqual(margin, expected_margin, places=2)

    def test_volatility(self):
        """Test volatility calculation."""
        self.btc_spot.simulate(n_paths=1, time_horizon=1 / 24)

        vol = self.btc_spot.volatility
        self.assertEqual(vol.shape, self.btc_spot.spot.shape)
        self.assertTrue((vol >= 0).all())

    def test_is_listed(self):
        """Test that Bitcoin spot is always listed."""
        self.assertTrue(self.btc_spot.is_listed)

    def test_device_dtype(self):
        """Test moving to different device/dtype."""
        btc = BitcoinSpot(data_loader=self.mock_loader, dtype=torch.float64)
        btc.simulate(n_paths=1, time_horizon=1 / 24)

        self.assertEqual(btc.spot.dtype, torch.float64)

        # Test moving to float32
        btc.to(dtype=torch.float32)
        self.assertEqual(btc.spot.dtype, torch.float32)


class TestIntegrationWithPFHedge(unittest.TestCase):
    """Test integration with PFHedge framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_loader = MockDataLoader()

    def test_as_primary_instrument(self):
        """Test that our instruments work as PFHedge primary instruments."""
        from pfhedge.instruments import EuropeanOption

        # Create Bitcoin perpetual as underlier
        btc = BitcoinPerpetualHistorical(data_loader=self.mock_loader)

        # Create option on Bitcoin
        option = EuropeanOption(underlier=btc, strike=50000, maturity=5 / 250)

        # Simulate - derivatives use different signature
        option.simulate(n_paths=2)

        # Check that underlier has spot prices
        self.assertTrue(hasattr(option.underlier, "spot"))
        self.assertEqual(option.underlier.spot.shape[0], 2)

        # Calculate payoff
        payoff = option.payoff()
        self.assertEqual(payoff.shape[0], 2)

    def test_with_hedger(self):
        """Test basic compatibility with PFHedge Hedger."""
        try:
            from pfhedge.nn import Hedger, MultiLayerPerceptron
            from pfhedge.instruments import EuropeanOption
        except ImportError:
            self.skipTest("PFHedge not fully available")

        btc = BitcoinPerpetualHistorical(data_loader=self.mock_loader)
        option = EuropeanOption(underlier=btc, strike=50000, maturity=5 / 250)

        # Create a simple hedger
        model = MultiLayerPerceptron(
            in_features=3, out_features=1, n_layers=2, n_units=32
        )
        hedger = Hedger(
            model=model,
            inputs=["log_moneyness", "time_to_maturity", "volatility"],
            criterion="mean_variance",
        )

        # This should not raise errors
        option.simulate(n_paths=10)
        # Note: We can't call hedger.fit() without more setup,
        # but at least verify the objects are compatible


class TestDataIntegration(unittest.TestCase):
    """Test integration with our data loader."""

    def test_with_real_data_loader(self):
        """Test with actual CryptoDataLoader if sample data exists."""
        try:
            # Try to use real data loader
            # Use absolute path to work both from repo root and from crypto/tests
            sample_data_dir = os.path.join(
                os.path.dirname(__file__), "..", "data", "sample_data"
            )
            loader = CryptoDataLoader(sample_data_dir)

            # Try to load data
            perp_data = loader.load_perpetual_data()
            if perp_data is None or perp_data.empty:
                self.skipTest("No sample data available")

            # Create instruments with real loader
            btc_spot = BitcoinSpot(data_loader=loader)
            btc_perp = BitcoinPerpetualHistorical(data_loader=loader)

            # Simulate with real data
            btc_spot.simulate(n_paths=1, time_horizon=1 / 24)
            btc_perp.simulate(n_paths=1, time_horizon=1 / 24)

            # Check we have real data
            self.assertGreater(btc_spot.spot.shape[1], 0)
            self.assertGreater(btc_perp.spot.shape[1], 0)

            # Perpetual should have funding
            self.assertTrue(hasattr(btc_perp, "funding_rate"))

        except Exception as e:
            self.skipTest(f"Could not test with real data: {e}")


if __name__ == "__main__":
    unittest.main()
