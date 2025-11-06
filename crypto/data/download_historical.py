import os
import time
import gzip
import requests
from io import BytesIO
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
import numpy as np

try:
    from deribit_client import DeribitClient, timestamp_to_ms, ms_to_timestamp
except ImportError:
    from .deribit_client import DeribitClient, timestamp_to_ms, ms_to_timestamp


class HistoricalDataDownloader:

    def __init__(self, data_dir: str = "data", testnet: bool = True):
        self.client = DeribitClient(testnet=testnet)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def download_perpetual_data(
        self,
        start_date: datetime,
        end_date: datetime,
        instrument: str = "BTC-PERPETUAL",
        save_to_parquet: bool = True,
    ) -> pd.DataFrame:
        print(
            f"Downloading {instrument} data from {start_date.date()} to {end_date.date()}"
        )

        # For this POC, we'll collect hourly snapshots of ticker data
        data_points = []
        current_time = start_date

        with tqdm(total=int((end_date - start_date).total_seconds() / 3600)) as pbar:
            while current_time < end_date:
                try:
                    # Get ticker data (current price, bid/ask, etc.)
                    ticker = self.client.get_ticker(instrument)

                    data_point = {
                        "timestamp": current_time,
                        "instrument": instrument,
                        "last_price": ticker.get("last_price"),
                        "bid_price": ticker.get("best_bid_price"),
                        "ask_price": ticker.get("best_ask_price"),
                        "bid_size": ticker.get("best_bid_amount"),
                        "ask_size": ticker.get("best_ask_amount"),
                        "mark_price": ticker.get("mark_price"),
                        "index_price": ticker.get("index_price"),
                        "funding_8h": ticker.get("funding_8h", 0),
                        "open_interest": ticker.get("open_interest", 0),
                        "volume_24h": ticker.get("stats", {}).get("volume", 0),
                        "price_change_24h": ticker.get("stats", {}).get(
                            "price_change", 0
                        ),
                    }
                    data_points.append(data_point)

                    # Rate limit: 1 request per second to be safe
                    time.sleep(1)

                except Exception as e:
                    print(f"Error fetching data at {current_time}: {e}")

                current_time += timedelta(hours=1)
                pbar.update(1)

        df = pd.DataFrame(data_points)

        if save_to_parquet and not df.empty:
            filename = f"{self.data_dir}/{instrument.replace('-', '_')}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
            df.to_parquet(filename, index=False)
            print(f"Saved {len(df)} records to {filename}")

        return df

    def download_options_data(
        self,
        start_date: datetime,
        end_date: datetime,
        currency: str = "BTC",
        maturity_filter: Optional[str] = None,
        save_to_parquet: bool = True,
    ) -> pd.DataFrame:
        print(
            f"Downloading {currency} options data from {start_date.date()} to {end_date.date()}"
        )

        # Get available option instruments
        instruments = self.client.get_instruments(currency=currency, kind="option")

        # Filter instruments if needed
        if maturity_filter:
            instruments = [
                inst
                for inst in instruments
                if maturity_filter in inst["instrument_name"]
            ]

        # Focus on ATM options for POC
        atm_instruments = []
        spot_price = self.client.get_ticker(f"{currency}-PERPETUAL")["last_price"]

        for inst in instruments:
            strike = inst.get("strike")
            if (
                strike and abs(strike - spot_price) / spot_price < 0.1
            ):  # Within 10% of spot
                atm_instruments.append(inst)

        print(f"Found {len(atm_instruments)} ATM options to download")

        options_data = []

        # Sample a few representative options to avoid rate limits
        selected_instruments = atm_instruments[:5]  # Limit to 5 for POC

        for inst in tqdm(selected_instruments, desc="Options"):
            instrument_name = inst["instrument_name"]

            try:
                # Get current ticker data
                ticker = self.client.get_ticker(instrument_name)

                option_data = {
                    "timestamp": end_date,  # Using end_date as snapshot time
                    "instrument": instrument_name,
                    "strike": inst.get("strike"),
                    "option_type": inst.get("option_type"),
                    "expiration": inst.get("expiration_timestamp"),
                    "last_price": ticker.get("last_price"),
                    "bid_price": ticker.get("best_bid_price"),
                    "ask_price": ticker.get("best_ask_price"),
                    "bid_iv": ticker.get("bid_iv"),
                    "ask_iv": ticker.get("ask_iv"),
                    "mark_iv": ticker.get("mark_iv"),
                    "mark_price": ticker.get("mark_price"),
                    "delta": ticker.get("greeks", {}).get("delta"),
                    "gamma": ticker.get("greeks", {}).get("gamma"),
                    "theta": ticker.get("greeks", {}).get("theta"),
                    "vega": ticker.get("greeks", {}).get("vega"),
                    "open_interest": ticker.get("open_interest", 0),
                    "volume_24h": ticker.get("stats", {}).get("volume", 0),
                }
                options_data.append(option_data)

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"Error fetching {instrument_name}: {e}")

        df = pd.DataFrame(options_data)

        if save_to_parquet and not df.empty:
            filename = f"{self.data_dir}/{currency}_options_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
            df.to_parquet(filename, index=False)
            print(f"Saved {len(df)} option records to {filename}")

        return df

    def download_spot_data(
        self,
        start_date: datetime,
        end_date: datetime,
        exchange: str = "coinbase",
        symbol: str = "BTC-USD",
        resample_freq: str = "8H",
        api_key: Optional[str] = None,
        save_to_parquet: bool = True,
    ) -> pd.DataFrame:
        print(
            f"Downloading {exchange}/{symbol} spot data from {start_date.date()} to {end_date.date()}"
        )
        print(f"Using Tardis CSV downloads (fast, efficient)")

        all_data = []
        current_date = start_date.date()
        end = end_date.date()

        total_days = (end - current_date).days + 1

        with tqdm(total=total_days, desc=f"{exchange} spot") as pbar:
            while current_date <= end:
                # Build CSV URL
                url = (
                    f"https://datasets.tardis.dev/v1/{exchange}/trades/"
                    f"{current_date.year}/{current_date.month:02d}/{current_date.day:02d}/"
                    f"{symbol}.csv.gz"
                )

                try:
                    # Download with API key if provided
                    headers = {}
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"

                    response = requests.get(url, headers=headers, timeout=120)

                    if response.status_code == 200:
                        # Decompress and read CSV
                        with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
                            df = pd.read_csv(f)

                        # Convert timestamp (microseconds to datetime)
                        df["timestamp"] = pd.to_datetime(
                            df["timestamp"], unit="us", utc=True
                        )

                        all_data.append(df)
                        pbar.set_postfix({"trades": len(df)})

                    elif response.status_code == 404:
                        pbar.write(f"  ⚠️  No data for {current_date}")
                    elif response.status_code == 401 or response.status_code == 402:
                        pbar.write(f"  ⚠️  Auth required for {current_date} (402/401)")
                        pbar.write(
                            f"     First day of month is free, or provide API key"
                        )
                    else:
                        pbar.write(
                            f"  ⚠️  HTTP {response.status_code} for {current_date}"
                        )

                except Exception as e:
                    pbar.write(f"  ⚠️  Error on {current_date}: {e}")

                current_date += timedelta(days=1)
                pbar.update(1)

        if not all_data:
            raise ValueError("No data downloaded. Check date range and API key.")

        # Combine all days
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Downloaded {len(combined):,} total trades")

        # Resample to OHLCV bars
        if resample_freq:
            print(f"Resampling to {resample_freq} bars...")
            df_indexed = combined.set_index("timestamp")

            ohlcv = (
                pd.DataFrame(
                    {
                        "open": df_indexed["price"].resample(resample_freq).first(),
                        "high": df_indexed["price"].resample(resample_freq).max(),
                        "low": df_indexed["price"].resample(resample_freq).min(),
                        "close": df_indexed["price"].resample(resample_freq).last(),
                        "volume": df_indexed["amount"].resample(resample_freq).sum(),
                        "trades": df_indexed["price"].resample(resample_freq).count(),
                    }
                )
                .dropna()
                .reset_index()
            )

            # Format for backtest system (match perpetual data structure)
            ohlcv["last_price"] = ohlcv["close"]
            ohlcv["open_price"] = ohlcv["open"]
            ohlcv["high_price"] = ohlcv["high"]
            ohlcv["low_price"] = ohlcv["low"]

            # Estimate bid/ask spread (0.01% = 1 basis point)
            spread = 0.0001
            ohlcv["bid_price"] = ohlcv["last_price"] * (1 - spread / 2)
            ohlcv["ask_price"] = ohlcv["last_price"] * (1 + spread / 2)

            # For spot, index price is same as last price
            ohlcv["index_price"] = ohlcv["last_price"]
            ohlcv["exchange"] = exchange

            combined = ohlcv
            print(f"Created {len(combined):,} {resample_freq} bars")

        if save_to_parquet and not combined.empty:
            filename = (
                f"{self.data_dir}/btc_spot_{resample_freq}_"
                f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.parquet"
            )
            combined.to_parquet(filename, index=False)
            print(f"✓ Saved to {filename}")
            print(
                f"  Price range: ${combined['last_price'].min():,.2f} - ${combined['last_price'].max():,.2f}"
            )

        return combined

    def download_sample_dataset(self, days_back: int = 7) -> Dict[str, pd.DataFrame]:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        print(f"Downloading sample dataset: {days_back} days of data")

        # Download perpetual data (reduced frequency for testing)
        print("\n1. Downloading BTC perpetual data...")
        # For testing, just get a few snapshots instead of hourly
        perpetual_data = []
        sample_times = pd.date_range(start_date, end_date, freq="6H")  # Every 6 hours

        for timestamp in tqdm(sample_times, desc="Perpetual snapshots"):
            try:
                ticker = self.client.get_ticker("BTC-PERPETUAL")
                data_point = {
                    "timestamp": timestamp,
                    "instrument": "BTC-PERPETUAL",
                    "last_price": ticker.get("last_price"),
                    "bid_price": ticker.get("best_bid_price"),
                    "ask_price": ticker.get("best_ask_price"),
                    "mark_price": ticker.get("mark_price"),
                    "index_price": ticker.get("index_price"),
                    "funding_8h": ticker.get("funding_8h", 0),
                }
                perpetual_data.append(data_point)
                time.sleep(1)
            except Exception as e:
                print(f"Error: {e}")

        perpetual_df = pd.DataFrame(perpetual_data)

        # Download options data (current snapshot)
        print("\n2. Downloading BTC options data...")
        options_df = self.download_options_data(
            start_date, end_date, save_to_parquet=False
        )

        # Save combined dataset
        if not perpetual_df.empty:
            perpetual_df.to_parquet(
                f"{self.data_dir}/sample_perpetual.parquet", index=False
            )
        if not options_df.empty:
            options_df.to_parquet(
                f"{self.data_dir}/sample_options.parquet", index=False
            )

        return {"perpetual": perpetual_df, "options": options_df}


def main():
    downloader = HistoricalDataDownloader(data_dir="sample_data", testnet=True)

    try:
        # Download 3 days of sample data
        data = downloader.download_sample_dataset(days_back=3)

        print(f"\n✅ Download complete!")
        print(f"Perpetual data: {len(data['perpetual'])} records")
        print(f"Options data: {len(data['options'])} records")

        # Display sample data
        if not data["perpetual"].empty:
            print(f"\nPerpetual data sample:")
            print(data["perpetual"].head())

        if not data["options"].empty:
            print(f"\nOptions data sample:")
            print(data["options"].head())

    except Exception as e:
        print(f"❌ Download failed: {e}")


if __name__ == "__main__":
    main()
