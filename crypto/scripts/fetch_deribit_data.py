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

Usage:
    # Fetch from Deribit (recent data)
    python fetch_deribit_data.py --start 2024-01-01 --end 2024-01-31 --output-dir data/historical

    # Fetch from Tardis (historical data)
    python fetch_deribit_data.py --data-source tardis --tardis-api-key YOUR_KEY \
        --start 2024-01-01 --end 2024-01-31 --output-dir data/historical
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
    except (ValueError, ImportError) as e:
        logger.error(f"Failed to create client: {e}")
        return 1

    # 1. Fetch perpetual trades
    trades_df = fetch_perpetual_trades(
        client, start_date, end_date, instrument=args.instrument
    )

    ohlc_df = pd.DataFrame()  # Initialize to avoid unbound variable
    if not trades_df.empty:
        # Save raw trades
        save_data(
            trades_df,
            output_dir / "raw",
            f"{args.instrument.lower()}_trades_{args.start}_{args.end}",
            format=args.format,
        )

        # Resample to OHLC
        ohlc_df = resample_trades_to_ohlc(trades_df, frequency=args.frequency)

        # Save OHLC data
        save_data(
            ohlc_df,
            output_dir,
            f"{args.instrument.lower()}_{args.frequency}_{args.start}_{args.end}",
            format=args.format,
        )

    # 2. Fetch funding rates
    funding_df = fetch_funding_rates(
        client, start_date, end_date, instrument=args.instrument
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
    if not trades_df.empty:
        print(f"\nSummary:")
        print(f"  Perpetual trades: {len(trades_df)}")
        print(f"  OHLC bars: {len(ohlc_df) if not ohlc_df.empty else 0}")
        print(f"  Funding records: {len(funding_df) if not funding_df.empty else 0}")
        print(
            f"  Date range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}"
        )
        print(
            f"  Price range: ${trades_df['price'].min():.2f} - ${trades_df['price'].max():.2f}"
        )

    return 0


if __name__ == "__main__":
    main()
