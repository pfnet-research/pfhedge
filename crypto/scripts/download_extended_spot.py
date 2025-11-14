#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
import gzip
from io import BytesIO
import pandas as pd
from tqdm import tqdm
import logging
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
TARDIS_API_KEY = os.getenv("TARDIS_API_KEY")


def download_btc_index_csv(api_key, start_date, end_date):
    """Download BTC index price data from TARDIS CSV API"""

    all_data = []
    current_date = start_date

    progress = tqdm(
        total=(end_date - start_date).days + 1, desc="Downloading BTC index data"
    )

    while current_date <= end_date:
        url = (
            f"https://datasets.tardis.dev/v1/deribit/index_price/"
            f"{current_date.year}/{current_date.month:02d}/{current_date.day:02d}/"
            f"btc_usd.csv.gz"
        )

        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.get(url, headers=headers, timeout=60)

            if response.status_code == 200:
                with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
                    df = pd.read_csv(f)

                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)

                all_data.append(df)
                logger.debug(f"Downloaded {len(df)} records for {current_date}")

            elif response.status_code == 404:
                logger.warning(f"No data for {current_date} (404)")
            else:
                logger.error(f"HTTP {response.status_code} for {current_date}")

        except Exception as e:
            logger.error(f"Error downloading {current_date}: {e}")

        current_date += timedelta(days=1)
        progress.update(1)

    progress.close()

    if not all_data:
        raise ValueError("No data downloaded!")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    return combined


def resample_to_8h(df):
    """Resample to 8-hour intervals"""
    df = df.set_index("timestamp")

    resampled = (
        df.resample("8h")
        .agg(
            {
                "price": "last",
            }
        )
        .reset_index()
    )

    resampled.columns = ["timestamp", "close"]

    resampled = resampled.dropna()

    return resampled


def main():
    data_dir = Path(__file__).parent.parent / "data" / "historical"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Downloading Extended BTC Spot Data ===")
    logger.info(f"TARDIS API Key: {'Found' if TARDIS_API_KEY else 'Not found'}")

    if not TARDIS_API_KEY:
        raise ValueError("TARDIS_API_KEY not found in .env file")

    start_date_2023 = datetime(2023, 1, 1).date()
    end_date_2024 = datetime(2024, 12, 31).date()

    logger.info(
        f"\n1. Downloading 2023-2024 data: {start_date_2023} to {end_date_2024}"
    )
    df_2023_2024 = download_btc_index_csv(
        TARDIS_API_KEY, start_date_2023, end_date_2024
    )
    logger.info(f"   Downloaded {len(df_2023_2024)} records")

    logger.info("\n2. Resampling to 8H intervals")
    df_2023_2024_8h = resample_to_8h(df_2023_2024)
    logger.info(f"   Resampled to {len(df_2023_2024_8h)} records")

    existing_file = data_dir / "btc_spot_8H_2025-01-01_2025-10-28.parquet"
    if existing_file.exists():
        logger.info(f"\n3. Loading existing 2025 data from {existing_file}")
        df_2025 = pd.read_parquet(existing_file)
        logger.info(f"   Loaded {len(df_2025)} records")

        if "timestamp" in df_2025.columns:
            df_2025["timestamp"] = pd.to_datetime(df_2025["timestamp"], utc=True)

        logger.info("\n4. Combining datasets")
        combined = pd.concat([df_2023_2024_8h, df_2025], ignore_index=True)
        combined = combined.sort_values("timestamp")
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.reset_index(drop=True)

        logger.info(f"   Combined: {len(combined)} records")
        logger.info(
            f"   Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}"
        )

        output_file = data_dir / "btc_spot_8H_2023-01-01_2025-10-28.parquet"
        combined.to_parquet(output_file, index=False)
        logger.info(f"\n✅ Saved to: {output_file}")
        logger.info(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")

        logger.info("\n📊 Summary:")
        logger.info(f"   Total records: {len(combined)}")
        logger.info(f"   Start: {combined['timestamp'].min()}")
        logger.info(f"   End: {combined['timestamp'].max()}")
        logger.info(
            f"   Duration: {(combined['timestamp'].max() - combined['timestamp'].min()).days} days"
        )

        logger.info("\n   Price statistics:")
        logger.info(f"   Min: ${combined['close'].min():,.2f}")
        logger.info(f"   Max: ${combined['close'].max():,.2f}")
        logger.info(f"   Mean: ${combined['close'].mean():,.2f}")

    else:
        logger.warning(f"\n⚠️  Existing 2025 data not found: {existing_file}")
        logger.info("   Saving 2023-2024 data only")

        output_file = data_dir / "btc_spot_8H_2023-01-01_2024-12-31.parquet"
        df_2023_2024_8h.to_parquet(output_file, index=False)
        logger.info(f"\n✅ Saved to: {output_file}")


if __name__ == "__main__":
    main()
