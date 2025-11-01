"""
Data loader for processing downloaded Bitcoin and options data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import warnings

warnings.filterwarnings("ignore")


class CryptoDataLoader:
    """Load and process cryptocurrency and options data."""

    def __init__(self, data_dir: str = "sample_data"):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        self.perpetual_data = None
        self.spot_data = None
        self.options_data = None
        self.funding_data = None

    def load_perpetual_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load perpetual contract data.

        Args:
            filename: Specific file to load, or None for default

        Returns:
            DataFrame with processed perpetual data
        """
        if filename is None:
            # Look for perpetual OHLC file (exclude funding files)
            files = list(self.data_dir.glob("*perpetual*.parquet"))
            # Filter out funding files
            files = [f for f in files if "funding" not in f.name.lower()]
            if not files:
                raise FileNotFoundError(
                    f"No perpetual data files found in {self.data_dir}"
                )
            filename = files[0]
        else:
            filename = self.data_dir / filename

        print(f"Loading perpetual data from {filename}")
        df = pd.read_parquet(filename)

        # Clean and process data
        df = self._process_perpetual_data(df)
        self.perpetual_data = df
        return df

    def load_spot_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load Bitcoin spot data.

        Args:
            filename: Specific file to load, or None for default

        Returns:
            DataFrame with processed spot data
        """
        if filename is None:
            # Look for spot data files
            files = list(self.data_dir.glob("*spot*.parquet"))
            if not files:
                raise FileNotFoundError(f"No spot data files found in {self.data_dir}")
            # Use the most recent file if multiple exist
            filename = max(files, key=lambda f: f.stat().st_mtime)
        else:
            filename = self.data_dir / filename

        print(f"Loading spot data from {filename}")
        df = pd.read_parquet(filename)

        # Clean and process data (same as perpetual since structure is identical)
        df = self._process_perpetual_data(df)
        self.spot_data = df
        return df

    def load_options_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load options data.

        Args:
            filename: Specific file to load, or None for default

        Returns:
            DataFrame with processed options data
        """
        if filename is None:
            # Look for sample file
            files = list(self.data_dir.glob("*options*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"No options data files found in {self.data_dir}"
                )
            filename = files[0]
        else:
            filename = self.data_dir / filename

        print(f"Loading options data from {filename}")
        df = pd.read_parquet(filename)

        # Clean and process data
        df = self._process_options_data(df)
        self.options_data = df
        return df

    def load_funding_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Load funding rate data.

        Args:
            filename: Specific file to load, or None for default

        Returns:
            DataFrame with processed funding rate data
        """
        if filename is None:
            # Look for funding file
            files = list(self.data_dir.glob("*funding*.parquet"))
            if not files:
                # Funding data is optional
                return pd.DataFrame()
            filename = files[0]
        else:
            filename = self.data_dir / filename

        print(f"Loading funding data from {filename}")
        df = pd.read_parquet(filename)

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

        self.funding_data = df
        return df

    def _process_perpetual_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process perpetual contract data."""
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Handle OHLC-only files: use 'close' as fallback for 'last_price'
        if "last_price" not in df.columns and "close" in df.columns:
            df["last_price"] = df["close"]

        # Calculate basic features
        if "last_price" in df.columns:
            df["returns"] = df["last_price"].pct_change()
            df["log_returns"] = np.log(df["last_price"] / df["last_price"].shift(1))

        # Calculate bid-ask spread (may not exist in OHLC data)
        if "bid_price" in df.columns and "ask_price" in df.columns:
            df["spread"] = df["ask_price"] - df["bid_price"]
            df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
            df["spread_pct"] = df["spread"] / df["mid_price"]

        # Calculate rolling volatility (if we have enough data)
        if len(df) > 5 and "log_returns" in df.columns:
            df["volatility_5"] = df["log_returns"].rolling(window=5).std() * np.sqrt(
                24
            )  # Annualized

        # Fill NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].ffill()

        return df

    def _process_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process options data."""
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Process expiration
        if "expiration" in df.columns:
            df["expiration"] = pd.to_datetime(df["expiration"], unit="ms", utc=True)
            # Ensure timestamp is also timezone-aware
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            df["time_to_expiry"] = (
                df["expiration"] - df["timestamp"]
            ).dt.total_seconds() / (365.25 * 24 * 3600)

        # Calculate moneyness
        if "strike" in df.columns and "last_price" in df.columns:
            # Note: This assumes we have current spot price; in practice we'd need to merge with perpetual data
            spot_price = df["last_price"].median()  # Rough approximation
            df["moneyness"] = df["strike"] / spot_price
            df["log_moneyness"] = np.log(df["moneyness"])

        # Calculate bid-ask spread for options
        if "bid_price" in df.columns and "ask_price" in df.columns:
            df["spread"] = df["ask_price"] - df["bid_price"]
            df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
            df["spread_pct"] = df["spread"] / df["mid_price"]

        # Process implied volatility
        iv_columns = ["bid_iv", "ask_iv", "mark_iv"]
        for col in iv_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Fill NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].ffill()

        return df

    def get_price_series(self, frequency: str = "5T") -> pd.DataFrame:
        """
        Get resampled price series for backtesting.

        Args:
            frequency: Pandas frequency string (e.g., '5T' for 5 minutes, '1H' for 1 hour)

        Returns:
            DataFrame with resampled price data
        """
        if self.perpetual_data is None:
            raise ValueError("Must load perpetual data first")

        df = self.perpetual_data.copy()

        # Set timestamp as index for resampling
        df = df.set_index("timestamp")

        # Determine which price columns exist (handle OHLC-only files)
        price_cols = []
        for col in ["last_price", "bid_price", "ask_price", "mid_price"]:
            if col in df.columns:
                price_cols.append(col)

        # If we have OHLC data but no bid/ask, just use last_price
        if not price_cols and "last_price" in df.columns:
            price_cols = ["last_price"]

        # Resample to specified frequency
        price_series = df[price_cols].resample(frequency).last()

        # Forward fill missing values
        price_series = price_series.ffill()

        # Calculate returns on resampled data
        price_series["returns"] = price_series["last_price"].pct_change()
        price_series["log_returns"] = np.log(
            price_series["last_price"] / price_series["last_price"].shift(1)
        )

        return price_series.reset_index()

    def get_options_for_expiry(
        self, expiry_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get options data for a specific expiry.

        Args:
            expiry_date: Target expiry date, or None for nearest expiry

        Returns:
            DataFrame with options for specified expiry
        """
        if self.options_data is None:
            raise ValueError("Must load options data first")

        df = self.options_data.copy()

        if expiry_date is None:
            # Find the nearest expiry
            current_time = df["timestamp"].max()
            df["days_to_expiry"] = (
                df["expiration"] - current_time
            ).dt.total_seconds() / (24 * 3600)
            df = df[df["days_to_expiry"] > 0]  # Only future expiries
            if df.empty:
                return df
            nearest_expiry = df.loc[df["days_to_expiry"].idxmin(), "expiration"]
            df = df[df["expiration"] == nearest_expiry]
        else:
            df = df[df["expiration"].dt.date == expiry_date.date()]

        return df

    def get_atm_option(self, option_type: str = "call") -> Optional[Dict]:
        """
        Get the most ATM option of specified type.

        Args:
            option_type: 'call' or 'put'

        Returns:
            Dictionary with option data, or None if not found
        """
        if self.options_data is None:
            raise ValueError("Must load options data first")

        df = self.options_data.copy()
        df = df[df["option_type"].str.lower() == option_type.lower()]

        if df.empty:
            return None

        # Find the option closest to ATM
        df["distance_from_atm"] = abs(df["log_moneyness"])
        atm_option = df.loc[df["distance_from_atm"].idxmin()]

        return atm_option.to_dict()

    def create_backtest_dataset(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "1H",
    ) -> Dict[str, pd.DataFrame]:
        """
        Create a clean dataset for backtesting.

        Args:
            start_date: Start date for dataset
            end_date: End date for dataset
            frequency: Resampling frequency

        Returns:
            Dictionary with 'prices' and 'options' DataFrames
        """
        # Get price series
        prices = self.get_price_series(frequency=frequency)

        # Filter by date if specified
        if start_date:
            prices = prices[prices["timestamp"] >= start_date]
        if end_date:
            prices = prices[prices["timestamp"] <= end_date]

        # Get options data
        options = (
            self.options_data.copy()
            if self.options_data is not None
            else pd.DataFrame()
        )

        # Filter options by date if specified
        if not options.empty:
            if start_date:
                options = options[options["timestamp"] >= start_date]
            if end_date:
                options = options[options["timestamp"] <= end_date]

        return {"prices": prices, "options": options}

    def summary(self) -> Dict:
        """Get summary statistics of loaded data."""
        summary = {}

        if self.perpetual_data is not None:
            perp = self.perpetual_data
            summary["perpetual"] = {
                "records": len(perp),
                "date_range": (perp["timestamp"].min(), perp["timestamp"].max()),
                "price_range": (perp["last_price"].min(), perp["last_price"].max()),
                "avg_spread_pct": (
                    perp["spread_pct"].mean() if "spread_pct" in perp.columns else None
                ),
            }

        if self.options_data is not None:
            opts = self.options_data
            summary["options"] = {
                "records": len(opts),
                "unique_strikes": (
                    opts["strike"].nunique() if "strike" in opts.columns else 0
                ),
                "call_count": (
                    len(opts[opts["option_type"] == "call"])
                    if "option_type" in opts.columns
                    else 0
                ),
                "put_count": (
                    len(opts[opts["option_type"] == "put"])
                    if "option_type" in opts.columns
                    else 0
                ),
                "avg_iv": opts["mark_iv"].mean() if "mark_iv" in opts.columns else None,
            }

        return summary


def test_loader():
    """Test the data loader."""
    print("Testing CryptoDataLoader...")

    loader = CryptoDataLoader("sample_data")

    try:
        # Load data
        perpetual_df = loader.load_perpetual_data()
        options_df = loader.load_options_data()

        print(f"Loaded {len(perpetual_df)} perpetual records")
        print(f"Loaded {len(options_df)} options records")

        # Test price series
        price_series = loader.get_price_series(frequency="2H")
        print(f"Resampled to {len(price_series)} 2-hour intervals")

        # Test ATM option
        atm_call = loader.get_atm_option("call")
        if atm_call:
            print(
                f"ATM call: strike={atm_call['strike']}, IV={atm_call.get('mark_iv', 'N/A')}"
            )

        # Test backtest dataset
        dataset = loader.create_backtest_dataset()
        print(
            f"Backtest dataset: {len(dataset['prices'])} price points, {len(dataset['options'])} options"
        )

        # Print summary
        summary = loader.summary()
        print("\nData Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")

        print("\n✅ Data loader test passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_loader()
