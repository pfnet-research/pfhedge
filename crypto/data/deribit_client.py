"""
Minimal Deribit API client for historical data retrieval.
"""

import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

try:
    from .base_client import MarketDataClient
except ImportError:
    from base_client import MarketDataClient


class DeribitClient(MarketDataClient):
    """Simple Deribit REST API client for historical data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
    ):
        """
        Initialize Deribit client.

        Args:
            api_key: API key (optional for public endpoints)
            api_secret: API secret (optional for public endpoints)
            testnet: Use testnet (True) or mainnet (False)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = (
            "https://test.deribit.com/api/v2"
            if testnet
            else "https://www.deribit.com/api/v2"
        )
        self.session = requests.Session()

    def _make_request(
        self, method: str, endpoint: str, params: Optional[Dict] = None
    ) -> Dict:
        """Make HTTP request to Deribit API."""
        url = f"{self.base_url}/{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params or {})
            else:
                response = self.session.post(url, json=params or {})

            response.raise_for_status()

            data = response.json()
            if data.get("error"):
                raise Exception(f"API Error: {data['error']}")

            return data["result"]

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    def get_instruments(
        self, currency: str = "BTC", kind: str = "option", **kwargs
    ) -> List[Dict]:
        """
        Get available instruments.

        Args:
            currency: Currency (BTC, ETH, etc.)
            kind: Instrument kind (option, future, spot)
            **kwargs: Additional parameters (e.g., expired=False)

        Returns:
            List of instrument data
        """
        params = {"currency": currency, "kind": kind}
        # Add any additional parameters
        params.update(kwargs)
        return self._make_request("GET", "public/get_instruments", params)

    def get_historical_trades(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
        count: int = 1000,
    ) -> List[Dict]:
        """
        Get historical trades for an instrument.

        Args:
            instrument_name: Name of instrument (e.g., "BTC-PERPETUAL", "BTC-25DEC23-42000-C")
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds
            count: Number of trades to fetch (max 1000)

        Returns:
            List of trade data
        """
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "count": count,
            "include_old": True,
            "sorting": "asc",
        }
        result = self._make_request(
            "GET", "public/get_last_trades_by_instrument_and_time", params
        )

        # Deribit API returns {"trades": [...], "has_more": bool}
        # Extract and return just the trades list for interface compatibility
        if isinstance(result, dict) and "trades" in result:
            return result["trades"]
        # Fallback: if API changes or returns list directly
        return result if isinstance(result, list) else []

    def get_recent_trades(self, instrument_name: str, count: int = 10) -> List[Dict]:
        """
        Get recent trades for an instrument (simpler endpoint).

        Args:
            instrument_name: Name of instrument
            count: Number of recent trades to fetch

        Returns:
            List of trade data
        """
        params = {"instrument_name": instrument_name, "count": count}
        return self._make_request("GET", "public/get_last_trades_by_instrument", params)

    def get_historical_volatility(self, currency: str = "BTC") -> List[Dict]:
        """
        Get historical volatility data.

        Args:
            currency: Currency to get volatility for

        Returns:
            List of volatility data points
        """
        params = {"currency": currency}
        return self._make_request("GET", "public/get_historical_volatility", params)

    def get_ticker(self, instrument_name: str, timestamp: Optional[int] = None) -> Dict:
        """
        Get current ticker data for an instrument.

        Args:
            instrument_name: Name of instrument
            timestamp: Optional timestamp in milliseconds (ignored - Deribit API returns current data)

        Returns:
            Ticker data
        """
        params = {"instrument_name": instrument_name}
        if timestamp is not None:
            # Deribit's public ticker endpoint doesn't support historical timestamp
            # Log a warning if timestamp is provided
            import logging

            logging.getLogger(__name__).warning(
                f"DeribitClient.get_ticker() does not support historical timestamp parameter. "
                f"Returning current ticker data instead."
            )
        return self._make_request("GET", "public/ticker", params)

    def get_order_book(self, instrument_name: str, depth: int = 5) -> Dict:
        """
        Get order book for an instrument.

        Args:
            instrument_name: Name of instrument
            depth: Order book depth

        Returns:
            Order book data
        """
        params = {"instrument_name": instrument_name, "depth": depth}
        return self._make_request("GET", "public/get_order_book", params)

    def get_funding_rate_history(
        self,
        instrument_name: str = "BTC-PERPETUAL",
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get funding rate history for perpetual contract.

        Args:
            instrument_name: Name of perpetual instrument
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds

        Returns:
            List of funding rate data
        """
        params = {"instrument_name": instrument_name}
        if start_timestamp:
            params["start_timestamp"] = start_timestamp
        if end_timestamp:
            params["end_timestamp"] = end_timestamp

        return self._make_request("GET", "public/get_funding_rate_history", params)

    def get_ohlc_candles(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
        resolution: str = "60",
    ) -> Dict:
        """
        Get OHLC candlestick data from Deribit TradingView API.

        Args:
            instrument_name: Name of instrument (e.g., BTC-PERPETUAL)
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds
            resolution: Candle resolution in minutes
                Supported: 1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1D

        Returns:
            Dict with keys: ticks, open, high, low, close, volume, cost, status
            Example:
            {
                "ticks": [1735689600000, 1735693200000, ...],
                "open": [93445.5, 94225.5, ...],
                "high": [94320.0, 94225.5, ...],
                "low": [93336.0, 93440.5, ...],
                "close": [94224.5, 93467.0, ...],
                "volume": [238.92, 162.77, ...],
                "cost": [...],
                "status": "ok"
            }
        """
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "resolution": resolution,
        }

        return self._make_request("GET", "public/get_tradingview_chart_data", params)


def timestamp_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def ms_to_timestamp(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
