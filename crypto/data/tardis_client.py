"""
Tardis.dev API client for historical cryptocurrency data.

This client provides the same interface as DeribitClient but fetches historical
data from Tardis.dev, which has comprehensive historical coverage since 2019-03-30.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
from dateutil.parser import parse
from tardis_client import TardisClient as TardisAPIClient, Channel

from .base_client import MarketDataClient
from .deribit_client import timestamp_to_ms, ms_to_timestamp

# Configure logging
logger = logging.getLogger(__name__)


class TardisClient(MarketDataClient):
    """Tardis.dev client with Deribit-compatible interface.

    This client fetches historical data from Tardis.dev and converts it to the
    same format as DeribitClient, allowing seamless switching between data sources.

    Features:
    - Historical data since 2019-03-30
    - Tick-by-tick trades, order book, quotes, funding rates
    - First day of each month free without API key
    - Full access with API key

    Args:
        api_key: Tardis.dev API key (optional for free access)
        testnet: Ignored (Tardis only has mainnet historical data)

    Example:
        >>> client = TardisClient(api_key="your_api_key")
        >>> trades = client.get_historical_trades(
        ...     "BTC-PERPETUAL",
        ...     timestamp_to_ms(datetime(2024, 10, 1)),
        ...     timestamp_to_ms(datetime(2024, 10, 2))
        ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        testnet: bool = False,
    ):
        """Initialize Tardis client.

        Args:
            api_key: Tardis.dev API key (optional for free monthly access)
            testnet: Ignored - Tardis only has mainnet data
        """
        self.api_key = api_key
        self._tardis_client = (
            TardisAPIClient(api_key=api_key) if api_key else TardisAPIClient()
        )

        if testnet:
            logger.warning(
                "TardisClient does not support testnet. "
                "Using mainnet historical data instead."
            )

        # Cache for instruments list (Tardis doesn't have real-time instruments API)
        self._instruments_cache: Optional[List[Dict]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)

    def get_instruments(
        self, currency: str = "BTC", kind: str = "option", **kwargs
    ) -> List[Dict]:
        """Get available instruments using Tardis Instruments Metadata API.

        This method fetches instruments from Tardis's comprehensive instrument
        database, which includes historical instruments and their availability periods.

        Note: Requires Tardis API key (free tier limited, paid plans recommended).

        Args:
            currency: Currency (BTC, ETH, etc.)
            kind: Instrument kind (option, future, spot)
            **kwargs: Additional parameters:
                - active: bool (default True) - filter by active status
                - expired: bool - if True, return expired instruments
                - expiry_date: datetime - filter by specific expiry date (for options)

        Returns:
            List of instrument data in Deribit-compatible format

        Examples:
            >>> # Get active BTC options
            >>> client.get_instruments("BTC", "option")

            >>> # Get BTC options expiring on Oct 29, 2024
            >>> from datetime import datetime
            >>> expiry = datetime(2024, 10, 29)
            >>> client.get_instruments("BTC", "option", expiry_date=expiry)
        """
        # Check cache (only if no specific filters)
        now = datetime.now(timezone.utc)

        if (
            self._instruments_cache is not None
            and self._cache_timestamp is not None
            and now - self._cache_timestamp < self._cache_ttl
            and not kwargs.get("expiry_date")  # Don't use cache for filtered queries
        ):
            return self._filter_instruments(self._instruments_cache, currency, kind)

        logger.info(f"Fetching instruments from Tardis API: {currency} {kind}")

        try:
            # Build filter for Tardis Instruments API
            filter_payload = {"type": kind, "baseCurrency": currency}

            # Handle active parameter
            if "expired" in kwargs:
                # If expired=False, only active instruments
                if not kwargs["expired"]:
                    filter_payload["active"] = True
            elif "active" in kwargs:
                filter_payload["active"] = kwargs["active"]
            else:
                # Default: only active instruments
                filter_payload["active"] = True

            # Encode filter
            encoded_filter = requests.utils.quote(json.dumps(filter_payload))
            url = (
                f"https://api.tardis.dev/v1/instruments/deribit?filter={encoded_filter}"
            )

            # Make request with API key (increase timeout for large queries)
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            tardis_instruments = response.json()

            # Convert Tardis format to Deribit format
            instruments = self._convert_tardis_instruments_to_deribit_format(
                tardis_instruments
            )

            # Apply client-side filtering by expiry_date
            if "expiry_date" in kwargs and kwargs["expiry_date"]:
                expiry_date = kwargs["expiry_date"]
                expiry_str = expiry_date.strftime("%d%b%y").upper()
                instruments = [
                    inst
                    for inst in instruments
                    if expiry_str in inst.get("instrument_name", "")
                ]
                logger.info(
                    f"Filtered to {len(instruments)} instruments for expiry {expiry_str}"
                )

            # Update cache (only for unfiltered queries)
            if not kwargs.get("expiry_date"):
                self._instruments_cache = instruments
                self._cache_timestamp = now

            logger.info(f"Found {len(instruments)} {kind}s for {currency}")
            return instruments

        except Exception as e:
            logger.error(f"Failed to fetch instruments from Tardis: {e}")
            # Fallback to Deribit public API
            logger.warning("Falling back to Deribit public API")
            return self._get_instruments_from_deribit_fallback(currency, kind, **kwargs)

    def _convert_tardis_instruments_to_deribit_format(
        self, tardis_instruments: List[Dict]
    ) -> List[Dict]:
        """Convert Tardis instrument format to Deribit format.

        Tardis format:
            {id, baseCurrency, quoteCurrency, type, strikePrice, optionType, expiry, ...}

        Deribit format:
            {instrument_name, kind, strike, option_type, expiration_timestamp, ...}

        Args:
            tardis_instruments: List of instruments in Tardis format

        Returns:
            List of instruments in Deribit format
        """
        deribit_instruments = []

        for inst in tardis_instruments:
            # Convert to Deribit format
            deribit_inst = {
                "instrument_name": inst["id"],
                "kind": inst["type"],
                "quote_currency": inst.get("baseCurrency"),
                "base_currency": inst.get("quoteCurrency"),
            }

            # Add option-specific fields
            if inst["type"] == "option":
                deribit_inst["strike"] = inst.get("strikePrice")
                deribit_inst["option_type"] = inst.get("optionType")

                # Convert expiry to timestamp
                if "expiry" in inst:
                    expiry_dt = parse(inst["expiry"])
                    deribit_inst["expiration_timestamp"] = int(
                        expiry_dt.timestamp() * 1000
                    )

            # Add availability info
            if "availableSince" in inst:
                deribit_inst["available_since"] = inst["availableSince"]

            # Add active status
            if "active" in inst:
                deribit_inst["is_active"] = inst["active"]

            deribit_instruments.append(deribit_inst)

        return deribit_instruments

    def _get_instruments_from_deribit_fallback(
        self, currency: str, kind: str, **kwargs
    ) -> List[Dict]:
        """Fallback to Deribit public API if Tardis fails."""
        url = "https://www.deribit.com/api/v2/public/get_instruments"
        params = {"currency": currency, "kind": kind}

        # Add expired parameter if provided
        if "expired" in kwargs:
            expired_val = kwargs["expired"]
            if isinstance(expired_val, bool):
                params["expired"] = str(expired_val).lower()
            else:
                params["expired"] = expired_val

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if data.get("error"):
            raise Exception(f"API Error: {data['error']}")

        return data["result"]

    def _filter_instruments(
        self, instruments: List[Dict], currency: str, kind: str
    ) -> List[Dict]:
        """Filter instruments by currency and kind."""
        return [
            inst
            for inst in instruments
            if inst.get("quote_currency") == currency and inst.get("kind") == kind
        ]

    def get_historical_trades(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
        count: int = 1000,
    ) -> List[Dict]:
        """Get historical trades for an instrument.

        Args:
            instrument_name: Name of instrument (e.g., "BTC-PERPETUAL")
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds
            count: Maximum number of trades (note: Tardis may return more)

        Returns:
            List of trade data in Deribit format
        """
        logger.info(
            f"Fetching historical trades for {instrument_name} "
            f"from {ms_to_timestamp(start_timestamp)} to {ms_to_timestamp(end_timestamp)}"
        )

        # Convert timestamps to dates
        # For intraday queries (same day), we need to expand to full day or next day
        start_dt = ms_to_timestamp(start_timestamp)
        end_dt = ms_to_timestamp(end_timestamp)

        from_date = start_dt.strftime("%Y-%m-%d")

        # If end is same day, use next day to ensure Tardis accepts the range
        if start_dt.date() == end_dt.date():
            to_date = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            to_date = end_dt.strftime("%Y-%m-%d")

        # Fetch trades using Tardis replay API
        trades = []

        try:
            # Run async replay in sync context
            trades = asyncio.run(
                self._replay_trades(
                    instrument_name,
                    from_date,
                    to_date,
                    start_timestamp,
                    end_timestamp,
                    count,
                )
            )

            logger.info(f"Fetched {len(trades)} trades")
            return trades

        except Exception as e:
            logger.error(f"Error fetching trades from Tardis: {e}")
            # Return result dict structure if API returned it
            if isinstance(e.args[0] if e.args else None, dict):
                result = e.args[0]
                if "trades" in result:
                    return result.get("trades", [])
            return []

    async def _replay_trades(
        self,
        instrument_name: str,
        from_date: str,
        to_date: str,
        start_ms: int,
        end_ms: int,
        max_count: int,
    ) -> List[Dict]:
        """Replay historical trades from Tardis (async).

        Args:
            instrument_name: Instrument name
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            start_ms: Start timestamp in milliseconds
            end_ms: End timestamp in milliseconds
            max_count: Maximum trades to return

        Returns:
            List of trades in Deribit format
        """
        trades = []

        # Tardis replay API
        messages = self._tardis_client.replay(
            exchange="deribit",
            from_date=from_date,
            to_date=to_date,
            filters=[
                Channel(name="trades", symbols=[instrument_name]),
            ],
        )

        async for local_timestamp, message in messages:
            # Tardis returns WebSocket messages in the format:
            # {"jsonrpc": "2.0", "method": "subscription", "params": {"channel": "...", "data": [...]}}
            # Extract trades from params.data
            if "params" not in message or "data" not in message["params"]:
                continue

            for trade_data in message["params"]["data"]:
                # Filter by timestamp range
                msg_timestamp = trade_data.get("timestamp", 0)

                if msg_timestamp < start_ms or msg_timestamp > end_ms:
                    continue

                # Convert to Deribit format
                trade = self._convert_trade_message(trade_data)
                trades.append(trade)

                # Limit count
                if len(trades) >= max_count:
                    return trades

        return trades

    def _convert_trade_message(self, msg: Dict) -> Dict:
        """Convert Tardis trade message to Deribit format.

        Tardis message structure (from Deribit WebSocket v2):
        {
            "timestamp": 1609459200000,
            "trade_id": "123456",
            "price": 29000.5,
            "amount": 100,
            "direction": "buy",
            "instrument_name": "BTC-PERPETUAL",
            ...
        }

        Args:
            msg: Tardis message

        Returns:
            Trade data in Deribit REST API format
        """
        # Tardis already provides data in Deribit WebSocket format
        # Just extract the fields we need
        return {
            "timestamp": msg.get("timestamp"),
            "trade_id": msg.get("trade_id"),
            "price": msg.get("price"),
            "amount": msg.get("amount"),
            "direction": msg.get("direction"),
            "instrument_name": msg.get("instrument_name"),
            "tick_direction": msg.get("tick_direction"),
            "iv": msg.get("iv"),  # For options
            "index_price": msg.get("index_price"),
        }

    def get_ticker(self, instrument_name: str, timestamp: Optional[int] = None) -> Dict:
        """Get ticker data for an instrument.

        Args:
            instrument_name: Name of instrument
            timestamp: Timestamp in milliseconds (required for Tardis)

        Returns:
            Ticker data in Deribit format

        Raises:
            ValueError: If timestamp is not provided
        """
        if timestamp is None:
            raise ValueError(
                "TardisClient requires a timestamp for ticker data. "
                "Use get_ticker(instrument_name, timestamp_to_ms(your_datetime))"
            )

        logger.info(
            f"Fetching ticker for {instrument_name} at {ms_to_timestamp(timestamp)}"
        )

        # Get ticker from trades around the timestamp
        # Use a small window (e.g., ±5 minutes)
        window_ms = 5 * 60 * 1000  # 5 minutes
        start_ms = timestamp - window_ms
        end_ms = timestamp + window_ms

        trades = self.get_historical_trades(
            instrument_name,
            start_ms,
            end_ms,
            count=100,
        )

        if not trades:
            raise Exception(
                f"No trades found for {instrument_name} at {ms_to_timestamp(timestamp)}"
            )

        # Reconstruct ticker from trades
        df = pd.DataFrame(trades)

        # Find trades closest to target timestamp
        df["time_diff"] = abs(df["timestamp"] - timestamp)
        df = df.sort_values("time_diff")

        # Get last trade price
        last_trade = df.iloc[0]
        last_price = last_trade["price"]

        # Estimate bid/ask from trade direction
        # This is approximate - real ticker would have actual order book data
        spread_pct = 0.001  # 0.1% spread estimate
        if last_trade["direction"] == "buy":
            best_ask_price = last_price
            best_bid_price = last_price * (1 - spread_pct)
        else:
            best_bid_price = last_price
            best_ask_price = last_price * (1 + spread_pct)

        return {
            "instrument_name": instrument_name,
            "timestamp": timestamp,
            "last_price": last_price,
            "best_bid_price": best_bid_price,
            "best_ask_price": best_ask_price,
            "mark_price": last_price,  # Approximate
            "index_price": last_trade.get("index_price", last_price),
            "last_trade_timestamp": last_trade["timestamp"],
            "estimated": True,  # Flag that this is reconstructed
        }

    def get_funding_rate_history(
        self,
        instrument_name: str = "BTC-PERPETUAL",
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
    ) -> List[Dict]:
        """Get funding rate history for perpetual contract using CSV downloads.

        This method uses Tardis CSV datasets which is much faster than WebSocket replay.
        Downloads derivative_ticker CSV files for each day and samples at 8-hour intervals.

        Args:
            instrument_name: Name of perpetual instrument
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds

        Returns:
            List of funding rate data (sampled at 8-hour intervals: 00:00, 08:00, 16:00 UTC)
        """
        if not start_timestamp or not end_timestamp:
            raise ValueError(
                "TardisClient requires start_timestamp and end_timestamp for funding rates"
            )

        logger.info(
            f"Fetching funding rates for {instrument_name} "
            f"from {ms_to_timestamp(start_timestamp)} to {ms_to_timestamp(end_timestamp)} "
            f"(using CSV download - fast)"
        )

        try:
            funding_rates = self._download_funding_csv(
                instrument_name,
                start_timestamp,
                end_timestamp,
            )

            logger.info(f"Fetched {len(funding_rates)} funding rate records")
            return funding_rates

        except Exception as e:
            logger.error(f"Error fetching funding rates: {e}")
            return []

    def _download_funding_csv(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
    ) -> List[Dict]:
        """Download funding rate data from Tardis CSV datasets.

        Downloads derivative_ticker CSV files for each day in the date range,
        then samples at 8-hour intervals (00:00, 08:00, 16:00 UTC).

        Args:
            instrument_name: Instrument name (e.g., BTC-PERPETUAL)
            start_timestamp: Start time in milliseconds
            end_timestamp: End time in milliseconds

        Returns:
            List of funding rate records
        """
        import requests
        import gzip
        from io import BytesIO
        import pandas as pd
        from datetime import timedelta

        start_dt = ms_to_timestamp(start_timestamp)
        end_dt = ms_to_timestamp(end_timestamp)

        all_data = []
        current_date = start_dt.date()
        end_date = end_dt.date()

        while current_date <= end_date:
            # Build CSV URL
            # Format: https://datasets.tardis.dev/v1/deribit/derivative_ticker/YYYY/MM/DD/SYMBOL.csv.gz
            url = (
                f"https://datasets.tardis.dev/v1/deribit/derivative_ticker/"
                f"{current_date.year}/{current_date.month:02d}/{current_date.day:02d}/"
                f"{instrument_name}.csv.gz"
            )

            logger.debug(f"Downloading CSV for {current_date}")

            try:
                # Download with API key
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = requests.get(url, headers=headers, timeout=60)

                if response.status_code == 200:
                    # Decompress and read CSV
                    with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
                        df = pd.read_csv(f)

                    # Convert timestamp (microseconds to datetime)
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], unit="us", utc=True
                    )

                    # Sample at 8-hour intervals
                    df["hour"] = df["timestamp"].dt.hour
                    df["minute"] = df["timestamp"].dt.minute

                    # Get records at funding times (00:00, 08:00, 16:00, within first minute)
                    funding_times = df[
                        (df["hour"].isin([0, 8, 16])) & (df["minute"] == 0)
                    ].copy()

                    # Take first record at each 8-hour interval
                    funding_times["date_hour"] = funding_times["timestamp"].dt.floor(
                        "8h"
                    )
                    funding_8h = (
                        funding_times.groupby("date_hour").first().reset_index()
                    )

                    all_data.append(funding_8h)
                    logger.debug(f"  Got {len(funding_8h)} 8-hour records")

                elif response.status_code == 404:
                    logger.warning(f"  No data for {current_date} (404)")
                else:
                    logger.error(f"  HTTP {response.status_code} for {current_date}")

            except Exception as e:
                logger.error(f"  Error downloading {current_date}: {e}")

            # Next day
            current_date += timedelta(days=1)

        if not all_data:
            return []

        # Combine all days
        combined = pd.concat(all_data, ignore_index=True)

        # Filter by exact timestamp range
        combined = combined[
            (combined["timestamp"] >= start_dt) & (combined["timestamp"] <= end_dt)
        ]

        # Convert to list of dicts (matching Deribit format)
        funding_records = []
        for _, row in combined.iterrows():
            funding_records.append(
                {
                    "timestamp": int(
                        row["timestamp"].timestamp() * 1000
                    ),  # Back to milliseconds
                    "instrument_name": instrument_name,
                    "interest_8h": row["funding_rate"],  # Tardis calls it funding_rate
                    "index_price": row["index_price"],
                }
            )

        return funding_records

    async def _replay_funding(
        self,
        instrument_name: str,
        from_date: str,
        to_date: str,
        start_ms: int,
        end_ms: int,
    ) -> List[Dict]:
        """Replay funding rate data from Tardis (async).

        Args:
            instrument_name: Instrument name
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            start_ms: Start timestamp in milliseconds
            end_ms: End timestamp in milliseconds

        Returns:
            List of funding rate records
        """
        funding_data = []

        # Tardis captures funding rate from "perpetual" channel
        messages = self._tardis_client.replay(
            exchange="deribit",
            from_date=from_date,
            to_date=to_date,
            filters=[
                Channel(name="perpetual", symbols=[instrument_name]),
            ],
        )

        async for local_timestamp, message in messages:
            # Tardis returns WebSocket messages in the format:
            # {"jsonrpc": "2.0", "method": "subscription", "params": {"channel": "...", "data": {...}}}
            # Extract funding data from params.data
            if "params" not in message or "data" not in message["params"]:
                continue

            funding_msg = message["params"]["data"]
            msg_timestamp = funding_msg.get("timestamp", 0)

            if msg_timestamp < start_ms or msg_timestamp > end_ms:
                continue

            # Extract funding rate info
            # Note: Perpetual channel provides continuous 'interest' field
            # To get 8-hour rates, we sample at 8-hour intervals (00:00, 08:00, 16:00 UTC)
            if "interest" in funding_msg:
                # Check if this timestamp is at an 8-hour funding interval
                # Funding times are 00:00, 08:00, 16:00 UTC daily
                from datetime import datetime, timezone

                msg_dt = datetime.fromtimestamp(msg_timestamp / 1000, tz=timezone.utc)
                hour = msg_dt.hour

                # Only record at funding times (with 10-second tolerance to avoid duplicates)
                # Take the first message at each funding interval
                if hour in [0, 8, 16] and msg_dt.minute == 0 and msg_dt.second < 10:
                    # Check if we already have a record for this funding time
                    # (to avoid multiple samples from the same 8-hour period)
                    funding_hour_key = (msg_dt.year, msg_dt.month, msg_dt.day, hour)
                    if not any(
                        (
                            datetime.fromtimestamp(
                                r["timestamp"] / 1000, tz=timezone.utc
                            ).year,
                            datetime.fromtimestamp(
                                r["timestamp"] / 1000, tz=timezone.utc
                            ).month,
                            datetime.fromtimestamp(
                                r["timestamp"] / 1000, tz=timezone.utc
                            ).day,
                            datetime.fromtimestamp(
                                r["timestamp"] / 1000, tz=timezone.utc
                            ).hour,
                        )
                        == funding_hour_key
                        for r in funding_data
                    ):
                        funding_record = {
                            "timestamp": msg_timestamp,
                            "instrument_name": instrument_name,
                            "interest_8h": funding_msg.get(
                                "interest", 0
                            ),  # Continuous interest rate
                            "index_price": funding_msg.get("index_price"),
                        }
                        funding_data.append(funding_record)

        return funding_data

    def get_recent_trades(self, instrument_name: str, count: int = 10) -> List[Dict]:
        """Get recent trades for an instrument.

        Note: Tardis is for historical data. This method fetches the most recent
        available historical trades (typically yesterday or last complete day).

        Args:
            instrument_name: Name of instrument
            count: Number of recent trades to fetch

        Returns:
            List of recent trade data
        """
        logger.warning(
            "TardisClient is for historical data. "
            "get_recent_trades() returns last available historical trades, not live data."
        )

        # Fetch yesterday's data
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        start_of_day = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        trades = self.get_historical_trades(
            instrument_name,
            timestamp_to_ms(start_of_day),
            timestamp_to_ms(end_of_day),
            count=count,
        )

        # Return last N trades
        return trades[-count:] if len(trades) > count else trades
