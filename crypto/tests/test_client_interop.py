"""
Integration tests for client interoperability.

These tests verify that DeribitClient and TardisClient return compatible
data structures and can be used interchangeably in real code.
"""

import unittest
from unittest.mock import patch, Mock
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.base_client import MarketDataClient
from data.deribit_client import DeribitClient, timestamp_to_ms
from data.tardis_client import TardisClient
from data.client_factory import create_client


class TestClientFactory(unittest.TestCase):
    """Test client factory function."""

    def test_create_deribit_client_testnet(self):
        """Test creating Deribit client for testnet."""
        client = create_client("deribit", testnet=True)
        self.assertIsInstance(client, DeribitClient)
        self.assertIsInstance(client, MarketDataClient)
        self.assertIn("test.deribit.com", client.base_url)

    def test_create_deribit_client_mainnet(self):
        """Test creating Deribit client for mainnet."""
        client = create_client("deribit", testnet=False)
        self.assertIsInstance(client, DeribitClient)
        self.assertIsInstance(client, MarketDataClient)
        self.assertIn("www.deribit.com", client.base_url)

    def test_create_tardis_client(self):
        """Test creating Tardis client."""
        with patch("data.tardis_client.TardisAPIClient"):
            client = create_client("tardis", tardis_api_key="test_key")
            self.assertIsInstance(client, TardisClient)
            self.assertIsInstance(client, MarketDataClient)

    def test_create_tardis_without_api_key_free_tier(self):
        """Test that creating Tardis client without API key works (free tier)."""
        with patch("data.tardis_client.TardisAPIClient"):
            with patch("data.client_factory.logger") as mock_logger:
                client = create_client("tardis", tardis_api_key=None)

                # Should succeed and create client
                self.assertIsInstance(client, TardisClient)
                self.assertIsInstance(client, MarketDataClient)

                # Should log warning about free tier
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                self.assertIn("free tier", warning_msg.lower())

    def test_invalid_data_source_raises(self):
        """Test that invalid data source raises error."""
        with self.assertRaises(ValueError) as context:
            create_client("invalid_source")

        self.assertIn("Unknown data source", str(context.exception))


class TestClientInterfaceCompatibility(unittest.TestCase):
    """Test that both clients have compatible interfaces."""

    def setUp(self):
        """Set up test clients."""
        self.deribit_client = DeribitClient(testnet=True)
        with patch("data.tardis_client.TardisAPIClient"):
            self.tardis_client = TardisClient(api_key="test")

    def test_both_implement_market_data_client(self):
        """Test that both clients implement MarketDataClient."""
        self.assertIsInstance(self.deribit_client, MarketDataClient)
        self.assertIsInstance(self.tardis_client, MarketDataClient)

    def test_both_have_get_instruments(self):
        """Test that both clients have get_instruments method."""
        self.assertTrue(hasattr(self.deribit_client, "get_instruments"))
        self.assertTrue(callable(getattr(self.deribit_client, "get_instruments")))

        self.assertTrue(hasattr(self.tardis_client, "get_instruments"))
        self.assertTrue(callable(getattr(self.tardis_client, "get_instruments")))

    def test_both_have_get_historical_trades(self):
        """Test that both clients have get_historical_trades method."""
        self.assertTrue(hasattr(self.deribit_client, "get_historical_trades"))
        self.assertTrue(callable(getattr(self.deribit_client, "get_historical_trades")))

        self.assertTrue(hasattr(self.tardis_client, "get_historical_trades"))
        self.assertTrue(callable(getattr(self.tardis_client, "get_historical_trades")))

    def test_both_have_get_ticker(self):
        """Test that both clients have get_ticker method."""
        self.assertTrue(hasattr(self.deribit_client, "get_ticker"))
        self.assertTrue(callable(getattr(self.deribit_client, "get_ticker")))

        self.assertTrue(hasattr(self.tardis_client, "get_ticker"))
        self.assertTrue(callable(getattr(self.tardis_client, "get_ticker")))

    def test_both_have_get_funding_rate_history(self):
        """Test that both clients have get_funding_rate_history method."""
        self.assertTrue(hasattr(self.deribit_client, "get_funding_rate_history"))
        self.assertTrue(
            callable(getattr(self.deribit_client, "get_funding_rate_history"))
        )

        self.assertTrue(hasattr(self.tardis_client, "get_funding_rate_history"))
        self.assertTrue(
            callable(getattr(self.tardis_client, "get_funding_rate_history"))
        )

    def test_both_have_get_recent_trades(self):
        """Test that both clients have get_recent_trades method."""
        self.assertTrue(hasattr(self.deribit_client, "get_recent_trades"))
        self.assertTrue(callable(getattr(self.deribit_client, "get_recent_trades")))

        self.assertTrue(hasattr(self.tardis_client, "get_recent_trades"))
        self.assertTrue(callable(getattr(self.tardis_client, "get_recent_trades")))


class TestDataFormatCompatibility(unittest.TestCase):
    """Test that both clients return compatible data formats."""

    @patch("data.deribit_client.requests.Session.get")
    def test_get_instruments_format(self, mock_get):
        """Test that get_instruments returns compatible format."""
        # Mock Deribit response
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [
                {
                    "instrument_name": "BTC-29DEC23-50000-C",
                    "kind": "option",
                    "strike": 50000,
                    "expiration_timestamp": 1703836800000,
                    "option_type": "call",
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        deribit_client = DeribitClient(testnet=True)
        result = deribit_client.get_instruments(currency="BTC", kind="option")

        # Verify format
        self.assertIsInstance(result, list)
        self.assertIn("instrument_name", result[0])
        self.assertIn("kind", result[0])
        self.assertIn("strike", result[0])

    @patch("data.deribit_client.requests.Session.get")
    def test_get_historical_trades_format(self, mock_get):
        """Test that get_historical_trades returns compatible format."""
        # Mock Deribit response
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "trades": [
                    {
                        "timestamp": 1609459200000,
                        "trade_id": "123",
                        "price": 29000.5,
                        "amount": 100,
                        "direction": "buy",
                    }
                ],
                "has_more": False,
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        deribit_client = DeribitClient(testnet=True)
        start = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end = datetime(2021, 1, 2, tzinfo=timezone.utc)

        result = deribit_client.get_historical_trades(
            "BTC-PERPETUAL",
            timestamp_to_ms(start),
            timestamp_to_ms(end),
        )

        # Verify format (should be list - API dict is now extracted by client)
        self.assertIsInstance(result, list)
        if result:
            self.assertIn("timestamp", result[0])
            self.assertIn("price", result[0])
            self.assertIn("amount", result[0])

    @patch("data.deribit_client.requests.Session.get")
    def test_get_ticker_format(self, mock_get):
        """Test that get_ticker returns compatible format."""
        # Mock Deribit response
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "instrument_name": "BTC-PERPETUAL",
                "last_price": 29000.0,
                "best_bid_price": 28999.5,
                "best_ask_price": 29000.5,
                "mark_price": 29000.0,
                "index_price": 29001.0,
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        deribit_client = DeribitClient(testnet=True)
        result = deribit_client.get_ticker("BTC-PERPETUAL")

        # Verify format
        self.assertIsInstance(result, dict)
        self.assertIn("instrument_name", result)
        self.assertIn("last_price", result)
        self.assertIn("best_bid_price", result)
        self.assertIn("best_ask_price", result)

    @patch("data.deribit_client.requests.Session.get")
    def test_get_funding_rate_history_format(self, mock_get):
        """Test that get_funding_rate_history returns compatible format."""
        # Mock Deribit response
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": [
                {
                    "timestamp": 1609459200000,
                    "index_name": "btc_usd",
                    "interest_8h": 0.0001,
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        deribit_client = DeribitClient(testnet=True)
        start = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end = datetime(2021, 1, 2, tzinfo=timezone.utc)

        result = deribit_client.get_funding_rate_history(
            "BTC-PERPETUAL",
            timestamp_to_ms(start),
            timestamp_to_ms(end),
        )

        # Verify format
        self.assertIsInstance(result, list)
        if result:
            self.assertIn("timestamp", result[0])


class TestPolymorphicUsage(unittest.TestCase):
    """Test that clients can be used polymorphically."""

    def test_function_accepts_both_clients(self):
        """Test that a function accepting MarketDataClient works with both."""

        def fetch_data(client: MarketDataClient, instrument: str):
            """Example function that uses client polymorphically."""
            # This should work with any MarketDataClient implementation
            return {
                "has_get_instruments": hasattr(client, "get_instruments"),
                "has_get_trades": hasattr(client, "get_historical_trades"),
                "has_get_ticker": hasattr(client, "get_ticker"),
            }

        # Test with DeribitClient
        deribit_client = DeribitClient(testnet=True)
        deribit_result = fetch_data(deribit_client, "BTC-PERPETUAL")

        self.assertTrue(deribit_result["has_get_instruments"])
        self.assertTrue(deribit_result["has_get_trades"])
        self.assertTrue(deribit_result["has_get_ticker"])

        # Test with TardisClient
        with patch("data.tardis_client.TardisAPIClient"):
            tardis_client = TardisClient(api_key="test")
            tardis_result = fetch_data(tardis_client, "BTC-PERPETUAL")

            self.assertTrue(tardis_result["has_get_instruments"])
            self.assertTrue(tardis_result["has_get_trades"])
            self.assertTrue(tardis_result["has_get_ticker"])


if __name__ == "__main__":
    unittest.main()
