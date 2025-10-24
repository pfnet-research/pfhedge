#!/usr/bin/env python3
"""
Fetch historical data from Deribit or Tardis.dev for realistic backtesting.

This script fetches:
1. BTC-PERPETUAL historical trades
2. Funding rate history
3. Options data for specific strikes/expiries

Data sources:
- deribit: Live Deribit API (limited to ~24h historical data)
- tardis: Tardis.dev historical data (2019-03-30 onwards, requires API key)

IMPORTANT - Funding Rates:
- Tardis provides continuous interest rates (updated every second)
- Deribit provides official 8-hour funding rates (published at 00:00, 08:00, 16:00 UTC)
- For accurate backtesting P&L, use --use-deribit-funding to fetch official rates

Recommended usage for backtesting:
    # Fetch perpetual trades from Tardis + official funding rates from Deribit
    python fetch_deribit_data.py \
        --data-source tardis \
        --use-deribit-funding \
        --start 2024-01-01 --end 2024-01-31 \
        --output-dir data/historical

Usage examples:
    # Deribit only (recent data, both trades and funding)
    python fetch_deribit_data.py \
        --start 2024-01-01 --end 2024-01-31 \
        --output-dir data/historical

    # Tardis trades + Tardis funding (sampled continuous rates)
    python fetch_deribit_data.py \
        --data-source tardis \
        --start 2024-01-01 --end 2024-01-31 \
        --output-dir data/historical

    # Tardis trades + Deribit funding (recommended for backtesting)
    python fetch_deribit_data.py \
        --data-source tardis \
        --use-deribit-funding \
        --start 2024-01-01 --end 2024-01-31 \
        --output-dir data/historical
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto.data.deribit_client import timestamp_to_ms, ms_to_timestamp
from crypto.data.base_client import MarketDataClient
from crypto.data.client_factory import create_client, add_client_args

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_perpetual_ohlc(
    client: MarketDataClient,
    start_date: datetime,
    end_date: datetime,
    instrument: str = "BTC-PERPETUAL",
    resolution: str = "60",
    target_frequency: str = "8H",
) -> pd.DataFrame:
    """
    Fetch OHLC candles from Deribit API and optionally resample.

    Deribit API has a ~5000 candle limit per request. This function automatically
    batches requests to fetch longer time periods.

    Args:
        client: Deribit client instance
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
        instrument: Instrument name (default: BTC-PERPETUAL)
        resolution: Candle resolution in minutes (default: 60)
        target_frequency: Target resampling frequency (default: 8H)
            Set to None to skip resampling

    Returns:
        DataFrame with OHLC data at target frequency
    """
    logger.info(
        f"Fetching {instrument} OHLC candles from {start_date} to {end_date} "
        f"(resolution: {resolution}min, target: {target_frequency})"
    )

    # Calculate batch size to stay under 5000 candle limit
    # For 60min resolution: 4000 hours = ~167 days per batch
    resolution_mins = int(resolution) if resolution != "1D" else 1440
    max_candles_per_batch = 4000  # Leave some margin
    batch_hours = (max_candles_per_batch * resolution_mins) // 60

    all_dfs = []
    current_start = start_date

    while current_start < end_date:
        # Calculate batch end
        current_end = min(current_start + timedelta(hours=batch_hours), end_date)

        logger.info(f"Fetching batch: {current_start} to {current_end}")

        start_ms = timestamp_to_ms(current_start)
        end_ms = timestamp_to_ms(current_end)

        try:
            # Fetch candles from Deribit
            result = client.get_ohlc_candles(
                instrument_name=instrument,
                start_timestamp=start_ms,
                end_timestamp=end_ms,
                resolution=resolution,
            )

            if result and "ticks" in result and result["ticks"]:
                # Convert to DataFrame
                df_batch = pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            result["ticks"], unit="ms", utc=True
                        ),
                        "open": result["open"],
                        "high": result["high"],
                        "low": result["low"],
                        "close": result["close"],
                        "volume": result["volume"],
                    }
                )
                all_dfs.append(df_batch)
                logger.info(f"Fetched {len(df_batch)} candles for this batch")
            else:
                logger.warning(f"No data for batch {current_start} to {current_end}")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error fetching batch {current_start} to {current_end}: {e}")

        current_start = current_end

    if not all_dfs:
        logger.warning("No OHLC data received")
        return pd.DataFrame()

    # Concatenate all batches
    df = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates (may occur at batch boundaries)
    df = (
        df.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Add last_price (required by historical data loader)
    df["last_price"] = df["close"]

    logger.info(f"Fetched {len(df)} total candles at {resolution}min resolution")

    # Resample if needed
    if target_frequency and target_frequency != f"{resolution}min":
        df = resample_ohlc(df, target_frequency)

    return df


def resample_ohlc(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample OHLC data to a different frequency.

    Args:
        df: DataFrame with timestamp, open, high, low, close, volume
        frequency: Target frequency (e.g., '8H', '1D')

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    logger.info(f"Resampling OHLC to {frequency}")

    # Set timestamp as index
    df = df.set_index("timestamp")

    # Resample OHLC
    resampled = df.resample(frequency).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "last_price": "last",
        }
    )

    # Drop any rows with NaN (incomplete periods at the end)
    resampled = resampled.dropna()

    # Reset index
    resampled = resampled.reset_index()

    logger.info(f"Resampled to {len(resampled)} {frequency} bars")

    return resampled


def fetch_perpetual_trades(
    client: MarketDataClient,
    start_date: datetime,
    end_date: datetime,
    instrument: str = "BTC-PERPETUAL",
    batch_hours: int = 24,
) -> pd.DataFrame:
    """
    Fetch historical trades for perpetual futures.

    Args:
        client: Deribit client instance
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
        instrument: Instrument name (default: BTC-PERPETUAL)
        batch_hours: Hours per batch (API has limits)

    Returns:
        DataFrame with trade data
    """
    logger.info(f"Fetching {instrument} trades from {start_date} to {end_date}")

    all_trades = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(hours=batch_hours), end_date)

        start_ms = timestamp_to_ms(current_start)
        end_ms = timestamp_to_ms(current_end)

        logger.debug(f"Fetching batch: {current_start} to {current_end}")

        try:
            trades = client.get_historical_trades(
                instrument_name=instrument,
                start_timestamp=start_ms,
                end_timestamp=end_ms,
                count=1000,
            )

            if trades:
                all_trades.extend(trades)
                logger.debug(f"Fetched {len(trades)} trades")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            # Continue with next batch

        current_start = current_end

    if not all_trades:
        logger.warning("No trades fetched")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_trades)

    # Convert timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Fetched {len(df)} total trades")

    return df


def fetch_funding_rates(
    client: MarketDataClient,
    start_date: datetime,
    end_date: datetime,
    instrument: str = "BTC-PERPETUAL",
) -> pd.DataFrame:
    """
    Fetch funding rate history.

    Args:
        client: Deribit client instance
        start_date: Start datetime (UTC)
        end_date: End datetime (UTC)
        instrument: Perpetual instrument name

    Returns:
        DataFrame with funding rate history
    """
    logger.info(f"Fetching funding rates for {instrument}")

    start_ms = timestamp_to_ms(start_date)
    end_ms = timestamp_to_ms(end_date)

    try:
        funding_data = client.get_funding_rate_history(
            instrument_name=instrument, start_timestamp=start_ms, end_timestamp=end_ms
        )

        if not funding_data:
            logger.warning("No funding rate data fetched")
            return pd.DataFrame()

        df = pd.DataFrame(funding_data)

        # Convert timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        logger.info(f"Fetched {len(df)} funding rate records")

        return df

    except Exception as e:
        logger.error(f"Error fetching funding rates: {e}")
        return pd.DataFrame()


def resample_trades_to_ohlc(
    trades_df: pd.DataFrame, frequency: str = "8H"
) -> pd.DataFrame:
    """
    Resample trades to OHLC format.

    Args:
        trades_df: DataFrame with trades (must have timestamp, price, amount columns)
        frequency: Resampling frequency (default: 8H for funding intervals)

    Returns:
        DataFrame with OHLC data
    """
    if trades_df.empty:
        return pd.DataFrame()

    logger.info(f"Resampling trades to {frequency} OHLC")

    # Set timestamp as index
    trades_df = trades_df.set_index("timestamp")

    # Resample to OHLC
    ohlc = trades_df["price"].resample(frequency).ohlc()

    # Add last_price (required by CryptoDataLoader) - same as close
    ohlc["last_price"] = ohlc["close"]

    # Add volume if available
    if "amount" in trades_df.columns:
        volume = trades_df["amount"].resample(frequency).sum()
        ohlc["volume"] = volume

    # Add trade count
    ohlc["trade_count"] = trades_df["price"].resample(frequency).count()

    # Forward fill any missing values
    ohlc = ohlc.fillna(method="ffill")

    # Reset index
    ohlc = ohlc.reset_index()

    logger.info(f"Resampled to {len(ohlc)} OHLC bars")

    return ohlc


def fetch_option_trades(
    client: MarketDataClient,
    option_name: str,
    sale_time: datetime,
    window_minutes: int = 30,
) -> pd.DataFrame:
    """
    Fetch option trades around a specific time.

    Args:
        client: Deribit client instance
        option_name: Option instrument name (e.g., "BTC-15NOV24-50000-C")
        sale_time: Target time for trades
        window_minutes: Minutes before/after to fetch

    Returns:
        DataFrame with option trades
    """
    logger.info(f"Fetching trades for {option_name} around {sale_time}")

    start_time = sale_time - timedelta(minutes=window_minutes)
    end_time = sale_time + timedelta(minutes=window_minutes)

    start_ms = timestamp_to_ms(start_time)
    end_ms = timestamp_to_ms(end_time)

    try:
        trades = client.get_historical_trades(
            instrument_name=option_name,
            start_timestamp=start_ms,
            end_timestamp=end_ms,
            count=1000,
        )

        if not trades:
            logger.warning(f"No trades found for {option_name}")
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        # Convert timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        logger.info(f"Fetched {len(df)} option trades")

        return df

    except Exception as e:
        logger.error(f"Error fetching option trades: {e}")
        return pd.DataFrame()


def save_data(
    df: pd.DataFrame, output_dir: Path, filename: str, format: str = "parquet"
):
    """Save DataFrame to file."""
    if df.empty:
        logger.warning(f"Skipping empty DataFrame: {filename}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        filepath = output_dir / f"{filename}.parquet"
        df.to_parquet(filepath, index=False)
    else:  # CSV
        filepath = output_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False)

    logger.info(f"Saved {len(df)} rows to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical data from Deribit or Tardis.dev",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Date range
    parser.add_argument(
        "--start", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="crypto/data/historical",
        help="Output directory for data files",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="BTC-PERPETUAL",
        help="Perpetual instrument to fetch",
    )
    parser.add_argument(
        "--frequency", type=str, default="8H", help="Resampling frequency (default: 8H)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format",
    )

    # Add common client arguments (data source, testnet, API keys)
    add_client_args(parser)

    # Add hybrid mode flag
    parser.add_argument(
        "--use-deribit-funding",
        action="store_true",
        help="Fetch official funding rates from Deribit API (recommended for backtesting). "
        "Only applies when --data-source=tardis. Provides accurate 8-hour funding rates instead of sampled continuous rates.",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Add one day to end date to include the full day
    end_date = end_date + timedelta(days=1)

    output_dir = Path(args.output_dir)

    logger.info(f"Fetching data from {start_date} to {end_date}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data source: {args.data_source}")
    if args.data_source == "tardis":
        logger.info(
            f"Tardis API key: {'***' + args.tardis_api_key[-4:] if args.tardis_api_key and len(args.tardis_api_key) > 4 else 'Not provided'}"
        )
    else:
        logger.info(f"Testnet: {args.testnet}")

    # Initialize client based on data source
    try:
        client = create_client(
            data_source=args.data_source,
            testnet=args.testnet,
            tardis_api_key=args.tardis_api_key,
        )

        # Create separate client for funding if hybrid mode is requested
        funding_client = None
        if args.use_deribit_funding and args.data_source == "tardis":
            logger.info("Hybrid mode: Using Deribit for official funding rates")
            from crypto.data.deribit_client import DeribitClient

            funding_client = DeribitClient(testnet=args.testnet)
        else:
            funding_client = client

    except (ValueError, ImportError) as e:
        logger.error(f"Failed to create client: {e}")
        return 1

    # 1. Fetch perpetual OHLC data directly from Deribit API
    ohlc_df = fetch_perpetual_ohlc(
        client,
        start_date,
        end_date,
        instrument=args.instrument,
        resolution="60",  # Fetch hourly candles
        target_frequency=args.frequency.replace(
            "H", "h"
        ),  # Resample to target (e.g., 8h)
    )

    if not ohlc_df.empty:
        # Save OHLC data
        save_data(
            ohlc_df,
            output_dir,
            f"{args.instrument.lower()}_{args.frequency}_{args.start}_{args.end}",
            format=args.format,
        )

    # 2. Fetch funding rates (using funding_client for hybrid mode support)
    funding_df = fetch_funding_rates(
        funding_client, start_date, end_date, instrument=args.instrument
    )

    if not funding_df.empty:
        save_data(
            funding_df,
            output_dir,
            f"{args.instrument.lower()}_funding_{args.start}_{args.end}",
            format=args.format,
        )

    logger.info("Data fetching complete!")

    # Print summary
    if not ohlc_df.empty:
        print(f"\nSummary:")
        print(f"  OHLC bars ({args.frequency}): {len(ohlc_df)}")
        print(f"  Funding records: {len(funding_df) if not funding_df.empty else 0}")
        print(
            f"  Date range: {ohlc_df['timestamp'].min()} to {ohlc_df['timestamp'].max()}"
        )
        print(
            f"  Price range: ${ohlc_df['low'].min():.2f} - ${ohlc_df['high'].max():.2f}"
        )

    return 0


if __name__ == "__main__":
    main()
