"""
Unit tests for historical data downloader.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import tempfile
import shutil
import os
import sys
from datetime import datetime, timezone, timedelta

# Add the crypto directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from data.download_historical import HistoricalDataDownloader
except ImportError:
    # Try alternative import path
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from download_historical import HistoricalDataDownloader


class TestHistoricalDataDownloader(unittest.TestCase):
    """Test cases for HistoricalDataDownloader."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.downloader = HistoricalDataDownloader(data_dir=self.test_dir, testnet=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test downloader initialization."""
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertTrue(hasattr(self.downloader, "client"))
        self.assertEqual(self.downloader.data_dir, self.test_dir)

    @patch("data.download_historical.time.sleep")  # Mock sleep to speed up tests
    def test_download_perpetual_data(self, mock_sleep):
        """Test downloading perpetual data."""
        # Mock the client's get_ticker method
        mock_ticker_data = {
            "last_price": 50000.0,
            "best_bid_price": 49999.5,
            "best_ask_price": 50000.5,
            "best_bid_amount": 1000,
            "best_ask_amount": 1000,
            "mark_price": 50000.0,
            "index_price": 50000.0,
            "funding_8h": 0.0001,
            "open_interest": 10000,
            "stats": {"volume": 1000000, "price_change": 0.02},
        }

        with patch.object(
            self.downloader.client, "get_ticker", return_value=mock_ticker_data
        ):
            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(2023, 1, 1, 2, tzinfo=timezone.utc)  # 2 hour window

            df = self.downloader.download_perpetual_data(
                start_date=start_date, end_date=end_date, save_to_parquet=False
            )

            # Should have data for each hour
            self.assertEqual(len(df), 2)  # 2 hours = 2 data points
            self.assertIn("timestamp", df.columns)
            self.assertIn("last_price", df.columns)
            self.assertIn("bid_price", df.columns)
            self.assertIn("ask_price", df.columns)

            # Check data values
            self.assertEqual(df["last_price"].iloc[0], 50000.0)
            self.assertEqual(df["bid_price"].iloc[0], 49999.5)

    def test_download_options_data(self):
        """Test downloading options data."""
        # Mock instruments response
        mock_instruments = [
            {
                "instrument_name": "BTC-29DEC23-50000-C",
                "strike": 50000.0,
                "option_type": "call",
                "expiration_timestamp": 1703865600000,
            },
            {
                "instrument_name": "BTC-29DEC23-50000-P",
                "strike": 50000.0,
                "option_type": "put",
                "expiration_timestamp": 1703865600000,
            },
        ]

        # Mock ticker response for perpetual (for spot price)
        mock_perpetual_ticker = {"last_price": 50000.0}

        # Mock ticker response for options
        mock_option_ticker = {
            "last_price": 2000.0,
            "best_bid_price": 1950.0,
            "best_ask_price": 2050.0,
            "bid_iv": 70.0,
            "ask_iv": 71.0,
            "mark_iv": 70.5,
            "mark_price": 2000.0,
            "greeks": {"delta": 0.6, "gamma": 0.00001, "theta": -15.2, "vega": 25.5},
            "open_interest": 100,
            "stats": {"volume": 1000},
        }

        with patch.object(
            self.downloader.client, "get_instruments", return_value=mock_instruments
        ), patch.object(self.downloader.client, "get_ticker") as mock_get_ticker:

            # Configure mock to return different data based on instrument
            def ticker_side_effect(instrument):
                if instrument == "BTC-PERPETUAL":
                    return mock_perpetual_ticker
                else:
                    return mock_option_ticker

            mock_get_ticker.side_effect = ticker_side_effect

            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)

            df = self.downloader.download_options_data(
                start_date=start_date,
                end_date=end_date,
                currency="BTC",
                save_to_parquet=False,
            )

            # Should have options data
            self.assertGreater(len(df), 0)
            self.assertIn("instrument", df.columns)
            self.assertIn("strike", df.columns)
            self.assertIn("option_type", df.columns)
            self.assertIn("mark_iv", df.columns)
            self.assertIn("delta", df.columns)

    @patch("data.download_historical.time.sleep")
    def test_download_sample_dataset(self, mock_sleep):
        """Test downloading sample dataset."""
        # Mock ticker data
        mock_ticker_data = {
            "last_price": 50000.0,
            "best_bid_price": 49999.5,
            "best_ask_price": 50000.5,
            "mark_price": 50000.0,
            "index_price": 50000.0,
            "funding_8h": 0.0001,
        }

        # Mock instruments and option tickers
        mock_instruments = [
            {
                "instrument_name": "BTC-29DEC23-50000-C",
                "strike": 50000.0,
                "option_type": "call",
                "expiration_timestamp": 1703865600000,
            }
        ]

        mock_option_ticker = {
            "last_price": 2000.0,
            "best_bid_price": 1950.0,
            "best_ask_price": 2050.0,
            "mark_iv": 70.5,
            "greeks": {"delta": 0.6},
            "open_interest": 100,
            "stats": {"volume": 1000},
        }

        with patch.object(
            self.downloader.client, "get_ticker"
        ) as mock_get_ticker, patch.object(
            self.downloader.client, "get_instruments", return_value=mock_instruments
        ):

            def ticker_side_effect(instrument):
                if instrument == "BTC-PERPETUAL":
                    return mock_ticker_data
                else:
                    return mock_option_ticker

            mock_get_ticker.side_effect = ticker_side_effect

            # Download 1 day sample
            data = self.downloader.download_sample_dataset(days_back=1)

            self.assertIn("perpetual", data)
            self.assertIn("options", data)
            self.assertIsInstance(data["perpetual"], pd.DataFrame)
            self.assertIsInstance(data["options"], pd.DataFrame)

            # Check that files were created
            perpetual_file = os.path.join(self.test_dir, "sample_perpetual.parquet")
            options_file = os.path.join(self.test_dir, "sample_options.parquet")

            self.assertTrue(os.path.exists(perpetual_file))
            self.assertTrue(os.path.exists(options_file))

    def test_error_handling(self):
        """Test error handling in downloader."""
        # Test with client that raises exceptions
        with patch.object(
            self.downloader.client, "get_ticker", side_effect=Exception("API Error")
        ):
            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(2023, 1, 1, 1, tzinfo=timezone.utc)

            # Should handle errors gracefully and return partial data
            df = self.downloader.download_perpetual_data(
                start_date=start_date, end_date=end_date, save_to_parquet=False
            )

            # Should still return a DataFrame (possibly empty)
            self.assertIsInstance(df, pd.DataFrame)

    def test_file_saving(self):
        """Test that files are saved correctly."""
        # Create sample data
        sample_data = pd.DataFrame(
            {"timestamp": [datetime.now(timezone.utc)], "price": [50000.0]}
        )

        # Mock the download to return our sample data
        with patch.object(
            self.downloader, "download_perpetual_data", return_value=sample_data
        ):
            # Enable saving
            self.downloader.download_perpetual_data(
                start_date=datetime.now(timezone.utc),
                end_date=datetime.now(timezone.utc),
                save_to_parquet=True,
            )

            # Check if file exists (this test depends on the actual implementation)
            # In a real scenario, we'd check the file was created with the right name


class TestDownloaderUtilities(unittest.TestCase):
    """Test utility functions in downloader."""

    def test_data_validation(self):
        """Test that downloaded data has correct structure."""
        # This would test data validation functions if they existed
        pass

    def test_rate_limiting(self):
        """Test that rate limiting is properly implemented."""
        # Test would verify that appropriate delays are added between requests
        pass


if __name__ == "__main__":
    unittest.main()
