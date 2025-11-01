"""
Unit tests for crypto data loader.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import tempfile
import shutil
import os
import sys

# Add the crypto directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.loader import CryptoDataLoader


class TestCryptoDataLoader(unittest.TestCase):
    """Test cases for CryptoDataLoader."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.loader = CryptoDataLoader(data_dir=self.test_dir)

        # Create sample perpetual data
        self.sample_perpetual_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-01-01", periods=10, freq="1H", tz="UTC"
                ),
                "instrument": ["BTC-PERPETUAL"] * 10,
                "last_price": [50000 + i * 100 for i in range(10)],
                "bid_price": [49999 + i * 100 for i in range(10)],
                "ask_price": [50001 + i * 100 for i in range(10)],
                "mark_price": [50000 + i * 100 for i in range(10)],
                "index_price": [50000 + i * 100 for i in range(10)],
                "funding_8h": [0.0001] * 10,
            }
        )

        # Create sample options data
        self.sample_options_data = pd.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)] * 3,
                "instrument": [
                    "BTC-29DEC23-50000-C",
                    "BTC-29DEC23-50000-P",
                    "BTC-29DEC23-51000-C",
                ],
                "strike": [50000.0, 50000.0, 51000.0],
                "option_type": ["call", "put", "call"],
                "expiration": [1703865600000] * 3,  # Some future timestamp
                "last_price": [2000.0, 1500.0, 1200.0],
                "bid_price": [1950.0, 1450.0, 1150.0],
                "ask_price": [2050.0, 1550.0, 1250.0],
                "mark_iv": [70.5, 71.2, 69.8],
                "delta": [0.6, -0.4, 0.45],
                "gamma": [0.00001, 0.00001, 0.000008],
                "theta": [-15.2, -12.8, -14.1],
                "vega": [25.5, 25.5, 23.2],
            }
        )

        # Save test data to parquet files
        self.perpetual_file = os.path.join(self.test_dir, "test_perpetual.parquet")
        self.options_file = os.path.join(self.test_dir, "test_options.parquet")

        self.sample_perpetual_data.to_parquet(self.perpetual_file, index=False)
        self.sample_options_data.to_parquet(self.options_file, index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_load_perpetual_data(self):
        """Test loading perpetual data."""
        df = self.loader.load_perpetual_data(filename="test_perpetual.parquet")

        self.assertEqual(len(df), 10)
        self.assertIn("returns", df.columns)
        self.assertIn("log_returns", df.columns)
        self.assertIn("spread", df.columns)
        self.assertIn("mid_price", df.columns)
        self.assertIn("spread_pct", df.columns)

        # Check calculated fields
        expected_spread = (
            self.sample_perpetual_data["ask_price"]
            - self.sample_perpetual_data["bid_price"]
        )
        pd.testing.assert_series_equal(df["spread"], expected_spread, check_names=False)

    def test_load_options_data(self):
        """Test loading options data."""
        df = self.loader.load_options_data(filename="test_options.parquet")

        self.assertEqual(len(df), 3)
        self.assertIn("time_to_expiry", df.columns)
        self.assertIn("moneyness", df.columns)
        self.assertIn("log_moneyness", df.columns)
        self.assertIn("spread", df.columns)

        # Check that time to expiry is calculated
        self.assertTrue(all(df["time_to_expiry"] > 0))

    def test_get_price_series(self):
        """Test price series resampling."""
        # Load data first
        self.loader.load_perpetual_data(filename="test_perpetual.parquet")

        # Get 2-hour resampled data
        price_series = self.loader.get_price_series(frequency="2h")

        self.assertGreater(len(price_series), 0)
        self.assertIn("returns", price_series.columns)
        self.assertIn("log_returns", price_series.columns)

    def test_get_atm_option(self):
        """Test finding ATM option."""
        # Load data first
        self.loader.load_options_data(filename="test_options.parquet")

        # Get ATM call
        atm_call = self.loader.get_atm_option("call")
        self.assertIsNotNone(atm_call)
        self.assertEqual(atm_call["option_type"], "call")

        # Get ATM put
        atm_put = self.loader.get_atm_option("put")
        self.assertIsNotNone(atm_put)
        self.assertEqual(atm_put["option_type"], "put")

    def test_create_backtest_dataset(self):
        """Test creating backtest dataset."""
        # Load both datasets
        self.loader.load_perpetual_data(filename="test_perpetual.parquet")
        self.loader.load_options_data(filename="test_options.parquet")

        # Create backtest dataset
        dataset = self.loader.create_backtest_dataset()

        self.assertIn("prices", dataset)
        self.assertIn("options", dataset)
        self.assertGreater(len(dataset["prices"]), 0)
        self.assertGreater(len(dataset["options"]), 0)

    def test_summary(self):
        """Test data summary generation."""
        # Load both datasets
        self.loader.load_perpetual_data(filename="test_perpetual.parquet")
        self.loader.load_options_data(filename="test_options.parquet")

        summary = self.loader.summary()

        self.assertIn("perpetual", summary)
        self.assertIn("options", summary)

        # Check perpetual summary
        perp_summary = summary["perpetual"]
        self.assertEqual(perp_summary["records"], 10)
        self.assertIn("date_range", perp_summary)
        self.assertIn("price_range", perp_summary)

        # Check options summary
        opts_summary = summary["options"]
        self.assertEqual(opts_summary["records"], 3)
        self.assertIn("unique_strikes", opts_summary)

    def test_process_perpetual_data_calculations(self):
        """Test that perpetual data processing calculations are correct."""
        self.loader.load_perpetual_data(filename="test_perpetual.parquet")
        df = self.loader.perpetual_data

        # Test returns calculation
        expected_returns = df["last_price"].pct_change()
        pd.testing.assert_series_equal(
            df["returns"], expected_returns, check_names=False
        )

        # Test spread calculation
        expected_spread = df["ask_price"] - df["bid_price"]
        pd.testing.assert_series_equal(df["spread"], expected_spread, check_names=False)

        # Test mid price calculation
        expected_mid = (df["bid_price"] + df["ask_price"]) / 2
        pd.testing.assert_series_equal(df["mid_price"], expected_mid, check_names=False)

    def test_empty_data_handling(self):
        """Test handling of empty data files."""
        # Create minimal parquet file (can't save completely empty DataFrame)
        empty_df = pd.DataFrame({"dummy": []})
        empty_file = os.path.join(self.test_dir, "empty.parquet")
        empty_df.to_parquet(empty_file, index=False)

        # Should handle empty data gracefully - may not raise exception, just return empty processed data
        result = self.loader.load_perpetual_data(filename="empty.parquet")
        self.assertIsInstance(result, pd.DataFrame)

    def test_file_not_found(self):
        """Test file not found error handling."""
        with self.assertRaises(FileNotFoundError):
            loader = CryptoDataLoader(data_dir="/nonexistent/path")
            loader.load_perpetual_data()


class TestDataProcessingFunctions(unittest.TestCase):
    """Test data processing utility functions."""

    def test_timezone_handling(self):
        """Test proper timezone handling in data processing."""
        # Create data with timezone-naive timestamps
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="1H"),
                "expiration": [1703865600000] * 5,  # milliseconds timestamp
                "price": [50000] * 5,
            }
        )

        temp_dir = tempfile.mkdtemp()
        try:
            file_path = os.path.join(temp_dir, "test.parquet")
            data.to_parquet(file_path, index=False)

            loader = CryptoDataLoader(data_dir=temp_dir)
            df = loader.load_options_data(filename="test.parquet")

            # Should have timezone-aware timestamps
            self.assertIsNotNone(df["timestamp"].dt.tz)
            self.assertIsNotNone(df["expiration"].dt.tz)

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
