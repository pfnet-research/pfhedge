"""
Unit tests for Deribit API client.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import sys
import os

# Add the crypto directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.deribit_client import DeribitClient, timestamp_to_ms, ms_to_timestamp


class TestDeribitClient(unittest.TestCase):
    """Test cases for DeribitClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = DeribitClient(testnet=True)

    def test_init_testnet(self):
        """Test client initialization with testnet."""
        client = DeribitClient(testnet=True)
        self.assertIn("test.deribit.com", client.base_url)

    def test_init_mainnet(self):
        """Test client initialization with mainnet."""
        client = DeribitClient(testnet=False)
        self.assertIn("www.deribit.com", client.base_url)

    @patch("data.deribit_client.requests.Session.get")
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {"result": {"test": "data"}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client._make_request("GET", "test/endpoint", {"param": "value"})

        self.assertEqual(result, {"test": "data"})
        mock_get.assert_called_once()

    @patch("data.deribit_client.requests.Session.get")
    def test_make_request_api_error(self, mock_get):
        """Test API error handling."""
        # Mock API error response
        mock_response = Mock()
        mock_response.json.return_value = {"error": {"message": "API Error"}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with self.assertRaises(Exception) as context:
            self.client._make_request("GET", "test/endpoint")

        self.assertIn("API Error", str(context.exception))

    @patch("data.deribit_client.requests.Session.get")
    def test_get_instruments(self, mock_get):
        """Test get_instruments method."""
        # Mock response with sample instruments
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [
                {"instrument_name": "BTC-PERPETUAL", "kind": "future"},
                {"instrument_name": "BTC-25DEC23-42000-C", "kind": "option"},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client.get_instruments(currency="BTC", kind="option")

        self.assertEqual(len(result), 2)
        self.assertIn("instrument_name", result[0])

    @patch("data.deribit_client.requests.Session.get")
    def test_get_ticker(self, mock_get):
        """Test get_ticker method."""
        # Mock ticker response
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "instrument_name": "BTC-PERPETUAL",
                "last_price": 50000.0,
                "best_bid_price": 49999.5,
                "best_ask_price": 50000.5,
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client.get_ticker("BTC-PERPETUAL")

        self.assertEqual(result["instrument_name"], "BTC-PERPETUAL")
        self.assertEqual(result["last_price"], 50000.0)

    @patch("data.deribit_client.requests.Session.get")
    def test_get_order_book(self, mock_get):
        """Test get_order_book method."""
        # Mock order book response
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "instrument_name": "BTC-PERPETUAL",
                "best_bid_price": 49999.5,
                "best_ask_price": 50000.5,
                "bids": [[49999.5, 1000]],
                "asks": [[50000.5, 1000]],
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client.get_order_book("BTC-PERPETUAL", depth=5)

        self.assertEqual(result["best_bid_price"], 49999.5)
        self.assertEqual(result["best_ask_price"], 50000.5)

    @patch("data.deribit_client.requests.Session.get")
    def test_get_recent_trades(self, mock_get):
        """Test get_recent_trades method."""
        # Mock trades response
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [
                {"trade_id": "123", "price": 50000.0, "amount": 1000},
                {"trade_id": "124", "price": 50001.0, "amount": 500},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client.get_recent_trades("BTC-PERPETUAL", count=10)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["price"], 50000.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_timestamp_to_ms(self):
        """Test timestamp to milliseconds conversion."""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ms = timestamp_to_ms(dt)

        self.assertEqual(
            ms, 1672574400000
        )  # Known timestamp for 2023-01-01 12:00:00 UTC

    def test_ms_to_timestamp(self):
        """Test milliseconds to timestamp conversion."""
        ms = 1672574400000  # 2023-01-01 12:00:00 UTC
        dt = ms_to_timestamp(ms)

        self.assertEqual(dt.year, 2023)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 1)
        self.assertEqual(dt.hour, 12)

    def test_timestamp_roundtrip(self):
        """Test timestamp conversion roundtrip."""
        original_dt = datetime(2023, 6, 15, 14, 30, 45, tzinfo=timezone.utc)
        ms = timestamp_to_ms(original_dt)
        converted_dt = ms_to_timestamp(ms)

        # Should be equal within microsecond precision
        self.assertEqual(
            original_dt.replace(microsecond=0), converted_dt.replace(microsecond=0)
        )


if __name__ == "__main__":
    unittest.main()
