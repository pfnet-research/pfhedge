"""
Unit tests for TardisClient.

These tests verify that TardisClient properly implements the MarketDataClient
interface and returns data in Deribit-compatible format.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.base_client import MarketDataClient
from data.tardis_client import TardisClient
from data.deribit_client import timestamp_to_ms, ms_to_timestamp


class TestTardisClient(unittest.TestCase):
    """Test cases for TardisClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key"
        with patch("data.tardis_client.TardisAPIClient"):
            self.client = TardisClient(api_key=self.api_key, testnet=False)

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        with patch("data.tardis_client.TardisAPIClient") as mock_tardis:
            client = TardisClient(api_key="my_key")
            mock_tardis.assert_called_once_with(api_key="my_key")
            self.assertEqual(client.api_key, "my_key")

    def test_init_without_api_key(self):
        """Test client initialization without API key (free access)."""
        with patch("data.tardis_client.TardisAPIClient") as mock_tardis:
            client = TardisClient(api_key=None)
            mock_tardis.assert_called_once_with()
            self.assertIsNone(client.api_key)

    def test_init_testnet_warning(self):
        """Test that testnet flag logs a warning."""
        with patch("data.tardis_client.TardisAPIClient"):
            with patch("data.tardis_client.logger") as mock_logger:
                TardisClient(api_key="key", testnet=True)
                mock_logger.warning.assert_called_once()

    def test_implements_market_data_client(self):
        """Test that TardisClient implements MarketDataClient interface."""
        self.assertIsInstance(self.client, MarketDataClient)

    @patch("requests.get")
    def test_get_instruments(self, mock_get):
        """Test get_instruments fetches from Tardis Instruments API."""
        # Mock Tardis Instruments API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": "BTC-29DEC23-50000-C",
                "type": "option",
                "baseCurrency": "BTC",
                "quoteCurrency": "BTC",
                "strikePrice": 50000,
                "optionType": "call",
                "expiry": "2023-12-29T08:00:00.000Z",
                "active": True,
            },
            {
                "id": "BTC-29DEC23-50000-P",
                "type": "option",
                "baseCurrency": "BTC",
                "quoteCurrency": "BTC",
                "strikePrice": 50000,
                "optionType": "put",
                "expiry": "2023-12-29T08:00:00.000Z",
                "active": True,
            },
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.client.get_instruments(currency="BTC", kind="option")

        # Verify API was called correctly (Tardis API)
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        # Should call Tardis API
        self.assertIn("api.tardis.dev", call_args[0][0])

        # Verify result format (converted to Deribit format)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIn("instrument_name", result[0])
        self.assertIn("strike", result[0])
        self.assertIn("option_type", result[0])
        self.assertEqual(result[0]["instrument_name"], "BTC-29DEC23-50000-C")
        self.assertEqual(result[0]["strike"], 50000)
        self.assertEqual(result[0]["option_type"], "call")

    @patch("data.tardis_client.asyncio.run")
    def test_get_historical_trades(self, mock_asyncio_run):
        """Test get_historical_trades converts timestamps and calls replay."""
        # Mock async function result
        mock_trades = [
            {
                "timestamp": 1609459200000,
                "trade_id": "123",
                "price": 29000.5,
                "amount": 10,
                "direction": "buy",
                "instrument_name": "BTC-PERPETUAL",
            }
        ]
        mock_asyncio_run.return_value = mock_trades

        start = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end = datetime(2021, 1, 2, tzinfo=timezone.utc)

        result = self.client.get_historical_trades(
            "BTC-PERPETUAL",
            timestamp_to_ms(start),
            timestamp_to_ms(end),
            count=100,
        )

        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()

        # Verify result structure
        self.assertEqual(result, mock_trades)
        self.assertIn("timestamp", result[0])
        self.assertIn("price", result[0])
        self.assertIn("amount", result[0])

    def test_convert_trade_message(self):
        """Test trade message conversion from Tardis to Deribit format."""
        tardis_msg = {
            "timestamp": 1609459200000,
            "trade_id": "ABC123",
            "price": 29000.5,
            "amount": 100,
            "direction": "buy",
            "instrument_name": "BTC-PERPETUAL",
            "tick_direction": 1,
            "index_price": 29001.0,
        }

        result = self.client._convert_trade_message(tardis_msg)

        # Verify all required fields are present
        self.assertEqual(result["timestamp"], 1609459200000)
        self.assertEqual(result["trade_id"], "ABC123")
        self.assertEqual(result["price"], 29000.5)
        self.assertEqual(result["amount"], 100)
        self.assertEqual(result["direction"], "buy")
        self.assertEqual(result["instrument_name"], "BTC-PERPETUAL")

    @patch("data.tardis_client.TardisClient.get_historical_trades")
    @patch("data.tardis_client.pd.DataFrame")
    def test_get_ticker(self, mock_df_class, mock_get_trades):
        """Test get_ticker reconstructs ticker from trades."""
        # Mock trades data
        mock_trades = [
            {
                "timestamp": 1609459200000,
                "price": 29000.0,
                "direction": "buy",
                "index_price": 29001.0,
            },
            {
                "timestamp": 1609459210000,
                "price": 29005.0,
                "direction": "sell",
                "index_price": 29003.0,
            },
        ]
        mock_get_trades.return_value = mock_trades

        # Mock DataFrame behavior
        mock_df = MagicMock()
        mock_df.iloc = [
            Mock(
                price=29005.0,
                direction="sell",
                index_price=29003.0,
                timestamp=1609459210000,
            )
        ]
        mock_df_class.return_value = mock_df

        timestamp = 1609459205000  # Middle timestamp

        result = self.client.get_ticker("BTC-PERPETUAL", timestamp=timestamp)

        # Verify ticker structure
        self.assertIn("instrument_name", result)
        self.assertIn("timestamp", result)
        self.assertIn("last_price", result)
        self.assertIn("best_bid_price", result)
        self.assertIn("best_ask_price", result)
        self.assertIn("mark_price", result)
        self.assertTrue(result.get("estimated"))

    def test_get_ticker_requires_timestamp(self):
        """Test that get_ticker raises error without timestamp."""
        with self.assertRaises(ValueError) as context:
            self.client.get_ticker("BTC-PERPETUAL", timestamp=None)

        self.assertIn("timestamp", str(context.exception).lower())

    @patch("data.tardis_client.TardisClient._download_funding_csv")
    def test_get_funding_rate_history(self, mock_download_csv):
        """Test get_funding_rate_history."""
        # Mock funding data from CSV download
        mock_funding = [
            {
                "timestamp": 1609459200000,
                "instrument_name": "BTC-PERPETUAL",
                "interest_8h": 0.0001,
                "index_price": 29000.0,
            }
        ]
        mock_download_csv.return_value = mock_funding

        start = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end = datetime(2021, 1, 2, tzinfo=timezone.utc)

        result = self.client.get_funding_rate_history(
            "BTC-PERPETUAL",
            timestamp_to_ms(start),
            timestamp_to_ms(end),
        )

        # Verify result structure
        self.assertEqual(result, mock_funding)
        self.assertIn("timestamp", result[0])
        self.assertIn("interest_8h", result[0])

    def test_get_funding_rate_requires_timestamps(self):
        """Test that get_funding_rate_history requires timestamps."""
        with self.assertRaises(ValueError):
            self.client.get_funding_rate_history("BTC-PERPETUAL", None, None)

    @patch("data.tardis_client.TardisClient.get_historical_trades")
    def test_get_recent_trades(self, mock_get_historical):
        """Test get_recent_trades returns recent historical data."""
        mock_trades = [{"timestamp": i, "price": 29000 + i} for i in range(20)]
        mock_get_historical.return_value = mock_trades

        result = self.client.get_recent_trades("BTC-PERPETUAL", count=10)

        # Should return last 10 trades
        self.assertEqual(len(result), 10)
        # Should be the last 10 from the mock data
        self.assertEqual(result, mock_trades[-10:])

    def test_instruments_cache(self):
        """Test that instruments list is cached."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            # Mock Tardis API response
            mock_response.json.return_value = [
                {
                    "id": "BTC-PERPETUAL",
                    "type": "option",
                    "baseCurrency": "BTC",
                    "quoteCurrency": "BTC",
                    "active": True,
                }
            ]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # First call
            result1 = self.client.get_instruments()
            # Second call (should use cache)
            result2 = self.client.get_instruments()

            # Should only make one API call
            self.assertEqual(mock_get.call_count, 1)
            self.assertEqual(result1, result2)


# TestTardisClientImportError class removed - tardis-client is now required


class TestTimestampConversion(unittest.TestCase):
    """Test timestamp conversion utilities."""

    def test_timestamp_to_ms(self):
        """Test datetime to milliseconds conversion."""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ms = timestamp_to_ms(dt)

        self.assertEqual(ms, 1672574400000)

    def test_ms_to_timestamp(self):
        """Test milliseconds to datetime conversion."""
        ms = 1672574400000
        dt = ms_to_timestamp(ms)

        self.assertEqual(dt.year, 2023)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 1)
        self.assertEqual(dt.hour, 12)

    def test_roundtrip_conversion(self):
        """Test timestamp conversion roundtrip."""
        original = datetime(2023, 6, 15, 14, 30, 45, tzinfo=timezone.utc)
        ms = timestamp_to_ms(original)
        converted = ms_to_timestamp(ms)

        # Should match within microsecond precision
        self.assertEqual(
            original.replace(microsecond=0), converted.replace(microsecond=0)
        )


if __name__ == "__main__":
    unittest.main()
