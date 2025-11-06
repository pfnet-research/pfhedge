from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING

import pandas as pd

from crypto.data import io_utils

if TYPE_CHECKING:
    from .config import BacktestConfig
    from crypto.data.loader import CryptoDataLoader

logger = logging.getLogger(__name__)


def load_for_backtesting(config: "BacktestConfig") -> "CryptoDataLoader":
    from crypto.data.loader import CryptoDataLoader

    data_dir = config.data_dir

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Tip: If using YAML config, relative paths are resolved relative to the config file.\n"
            f"      If creating config programmatically, use absolute paths or resolve manually."
        )

    logger.info(f"Loading historical data from {data_dir}...")

    # Step 1: Load raw price data
    raw_df = _load_price_data(config)

    # Step 2: Process the raw data
    processed_df = io_utils.process_price_data(raw_df)

    # Step 3: Resample to configured frequency
    resampled_df = _resample_data(config, processed_df)

    # Step 4: Filter by date range
    start_date = pd.to_datetime(config.start_date)
    end_date = pd.to_datetime(config.end_date)

    if start_date.tz is None:
        start_date = start_date.tz_localize("UTC")
    if end_date.tz is None:
        end_date = end_date.tz_localize("UTC")

    logger.info(f"Filtering data from {config.start_date} to {config.end_date}...")

    filtered_df = io_utils.filter_by_date_range(resampled_df, start_date, end_date)

    if filtered_df.empty:
        raise ValueError(
            f"No data found in date range [{config.start_date}, {config.end_date}]. "
            f"Available data range: [{resampled_df['timestamp'].min()}, {resampled_df['timestamp'].max()}]"
        )

    logger.info(f"✅ Filtered to {len(filtered_df):,} records in date range")

    # Step 5: Load options data (optional)
    options_df = _load_options_data(config, start_date, end_date)

    # Step 6: Load and merge funding data (if perpetual)
    if config.underlying_type == "perpetual":
        funding_df = _load_funding_data(config, start_date, end_date)
        if funding_df is not None:
            filtered_df = io_utils.merge_funding_rates(filtered_df, funding_df)
            resampled_df = io_utils.merge_funding_rates(resampled_df, funding_df)
    else:
        logger.info("ℹ️  Skipping funding data (spot instrument has no funding rates)")

    # Step 7: Create and populate CryptoDataLoader
    loader = CryptoDataLoader(data_dir)
    loader.perpetual_data = filtered_df
    loader.perpetual_data_full = resampled_df
    if options_df is not None:
        loader.options_data = options_df

    logger.info("✅ Data loading complete\n")

    return loader


def _load_price_data(config: "BacktestConfig") -> pd.DataFrame:
    data_file = config.data_file
    underlying_type = config.underlying_type

    logger.info(f"Loading data from specified file: {data_file}")

    try:
        df = io_utils.load_price_data(config.data_dir, data_file, underlying_type)
        logger.info(f"✅ Loaded {len(df):,} raw records")
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No data found in {config.data_dir}. "
            f"Expected files matching '*spot*.parquet' or '*perpetual*.parquet'. Error: {e}"
        )


def _resample_data(config: "BacktestConfig", df: pd.DataFrame) -> pd.DataFrame:
    dt_hours = config.dt_hours

    if dt_hours == int(dt_hours):
        frequency = f"{int(dt_hours)}H"
    else:
        dt_minutes = int(round(dt_hours * 60))
        frequency = f"{dt_minutes}T"

    logger.info(f"Resampling to {frequency} frequency...")

    try:
        resampled_df = io_utils.resample_ohlc(df, frequency)
        logger.info(
            f"✅ Resampled to {len(resampled_df):,} records at {frequency} intervals"
        )
    except Exception as e:
        raise ValueError(f"Failed to resample data at frequency '{frequency}': {e}")

    resampled_df = io_utils.normalize_timestamps(resampled_df)

    return resampled_df


def _load_options_data(
    config: "BacktestConfig",
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame | None:
    options_df = io_utils.load_options_data(config.data_dir)

    if options_df is None:
        logger.warning("⚠️  No options data found (this is okay for basic backtesting)")
        return None

    if not options_df.empty and "timestamp" in options_df.columns:
        options_df = io_utils.normalize_timestamps(options_df)
        options_df = io_utils.filter_by_date_range(options_df, start_date, end_date)

    logger.info(f"✅ Loaded {len(options_df):,} options records in date range")
    return options_df


def _load_funding_data(
    config: "BacktestConfig",
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame | None:
    funding_df = io_utils.load_funding_data(config.data_dir)

    if funding_df is None:
        logger.warning(
            "⚠️  No funding data found (this is okay, but funding costs won't be applied)"
        )
        return None

    if not funding_df.empty and "timestamp" in funding_df.columns:
        funding_df = io_utils.normalize_timestamps(funding_df)

        funding_in_range = io_utils.filter_by_date_range(
            funding_df, start_date, end_date
        )

        logger.info(
            f"✅ Loaded {len(funding_in_range):,} funding rate records in date range"
        )

        return funding_df

    return None


# Backward compatibility: Keep BacktestDataLoader as a wrapper
class BacktestDataLoader:

    def __init__(self, config: "BacktestConfig"):
        import warnings

        warnings.warn(
            "BacktestDataLoader is deprecated. Use load_for_backtesting(config) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config

    def load_and_prepare_data(self) -> "CryptoDataLoader":
        return load_for_backtesting(self.config)

    @staticmethod
    def _normalize_timestamps(
        df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        return io_utils.normalize_timestamps(df, timestamp_col)

    @staticmethod
    def _merge_funding_rates(
        price_df: pd.DataFrame, funding_df: pd.DataFrame
    ) -> pd.DataFrame:
        return io_utils.merge_funding_rates(price_df, funding_df)
