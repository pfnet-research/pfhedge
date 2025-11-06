import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

try:
    from .base_client import MarketDataClient
except ImportError:
    from base_client import MarketDataClient


class DeribitClient(MarketDataClient):

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
    ):
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
        params = {"instrument_name": instrument_name, "count": count}
        return self._make_request("GET", "public/get_last_trades_by_instrument", params)

    def get_historical_volatility(self, currency: str = "BTC") -> List[Dict]:
        params = {"currency": currency}
        return self._make_request("GET", "public/get_historical_volatility", params)

    def get_ticker(self, instrument_name: str, timestamp: Optional[int] = None) -> Dict:
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
        params = {"instrument_name": instrument_name, "depth": depth}
        return self._make_request("GET", "public/get_order_book", params)

    def get_funding_rate_history(
        self,
        instrument_name: str = "BTC-PERPETUAL",
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
    ) -> List[Dict]:
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
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "resolution": resolution,
        }

        return self._make_request("GET", "public/get_tradingview_chart_data", params)


def timestamp_to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def ms_to_timestamp(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
