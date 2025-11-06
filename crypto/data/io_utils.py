from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def load_price_data(
    data_dir: str, filename: str | None, underlying_type: str
) -> pd.DataFrame:
    data_path = Path(data_dir)

    if filename is None:
        # Auto-detect file based on underlying_type
        if underlying_type == "spot":
            files = list(data_path.glob("*spot*.parquet"))
        else:
            files = list(data_path.glob("*perpetual*.parquet"))
            # Filter out funding files
            files = [f for f in files if "funding" not in f.name.lower()]

        if not files:
            raise FileNotFoundError(
                f"No {underlying_type} data files found in {data_dir}"
            )

        # Use the most recent file if multiple exist
        filepath = max(files, key=lambda f: f.stat().st_mtime)
    else:
        filepath = data_path / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_parquet(filepath)

    if df.empty:
        raise ValueError(f"Empty data file: {filepath}")

    return df


def process_price_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    if "last_price" not in df.columns and "close" in df.columns:
        df["last_price"] = df["close"]

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].ffill()

    return df


def resample_ohlc(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("timestamp")

    price_cols = []
    for col in ["last_price", "bid_price", "ask_price", "mid_price"]:
        if col in df.columns:
            price_cols.append(col)

    if not price_cols and "last_price" in df.columns:
        price_cols = ["last_price"]

    resampled = df[price_cols].resample(frequency).last()
    resampled = resampled.ffill()

    resampled["returns"] = resampled["last_price"].pct_change()
    resampled["log_returns"] = np.log(
        resampled["last_price"] / resampled["last_price"].shift(1)
    )

    return resampled.reset_index()


def filter_by_date_range(
    df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
    return df[mask].reset_index(drop=True)


def load_options_data(data_dir: str) -> pd.DataFrame | None:
    data_path = Path(data_dir)
    files = list(data_path.glob("*options*.parquet"))

    if not files:
        return None

    # Use most recent file if multiple exist
    filepath = max(files, key=lambda f: f.stat().st_mtime)
    df = pd.read_parquet(filepath)

    if df.empty:
        return None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def load_funding_data(data_dir: str) -> pd.DataFrame | None:
    data_path = Path(data_dir)
    files = list(data_path.glob("*funding*.parquet"))

    if not files:
        return None

    # Use most recent file if multiple exist
    filepath = max(files, key=lambda f: f.stat().st_mtime)
    df = pd.read_parquet(filepath)

    if df.empty:
        return None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def merge_funding_rates(
    price_df: pd.DataFrame, funding_df: pd.DataFrame
) -> pd.DataFrame:
    merged = price_df.merge(
        funding_df[["timestamp", "interest_8h"]], on="timestamp", how="left"
    )

    if "interest_8h" in merged.columns:
        merged["funding_rate"] = merged["interest_8h"]

    if "funding_rate" in merged.columns:
        merged["funding_rate"] = merged["funding_rate"].ffill().bfill()

    return merged


def normalize_timestamps(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        return df

    df = df.copy()

    # Ensure column is datetime type
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    if df[timestamp_col].dt.tz is None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
    elif str(df[timestamp_col].dt.tz) != "UTC":
        df[timestamp_col] = df[timestamp_col].dt.tz_convert("UTC")

    return df
