from typing import Optional, Tuple
import pandas as pd
import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar

from .bitcoin_perpetual_base import BitcoinPerpetualBase
from .volatility_mixin import VolatilityMixin


# Columns to rescale when adjusting spot prices (price-like)
PRICE_COLUMNS = [
    "last_price",  # Spot price
    "index_price",  # Index/oracle price
    "mark_price",  # Mark price
    "best_bid_price",  # Level 1 bid
    "best_ask_price",  # Level 1 ask
    "bid_price",  # Legacy bid
    "ask_price",  # Legacy ask
]

# Columns to preserve unchanged (rates, quantities, bps)
PRESERVE_COLUMNS = [
    "funding_rate",
    "interest_8h",
    "volume",
    "amount",
    "trades_count",
]


class BitcoinPerpetualHistorical(VolatilityMixin, BitcoinPerpetualBase):

    def __init__(
        self,
        data_loader: object,  # Required for historical data
        cost: float = 0.0006,
        dt: float = 1 / 24 / 12,
        leverage: float = 20.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        constant_volatility: Optional[
            float
        ] = None,  # NEW: Use constant vol for backtest
        volatility_window: int = 20,  # Rolling window for realized vol (0 = expanding)
    ) -> None:
        super().__init__(
            cost=cost, dt=dt, leverage=leverage, dtype=dtype, device=device
        )

        if data_loader is None:
            raise ValueError("data_loader is required for BitcoinPerpetualHistorical")

        self.data_loader = data_loader
        self.constant_volatility = (
            constant_volatility  # NEW: Store for use in volatility property
        )
        self.volatility_window = volatility_window

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        # Load perpetual data with funding rates
        # Prefer cached data (which may have funding merged) over reloading from disk
        if (
            self.data_loader.perpetual_data is not None
            and not self.data_loader.perpetual_data.empty
        ):
            perpetual_df = self.data_loader.perpetual_data
        else:
            perpetual_df = self.data_loader.load_perpetual_data()

        if perpetual_df is None or perpetual_df.empty:
            raise ValueError("No perpetual data available in data_loader")

        # Ensure required columns exist
        required_cols = ["timestamp", "last_price"]
        if not all(col in perpetual_df.columns for col in required_cols):
            raise ValueError(f"Perpetual data must contain: {required_cols}")

        # Calculate number of steps needed
        import math

        n_steps = math.ceil(time_horizon / self.dt + 1)

        # Limit data to required time horizon
        data = perpetual_df.head(n_steps).copy()
        actual_steps = len(data)

        if actual_steps < n_steps:
            import warnings

            warnings.warn(
                f"Requested {n_steps} steps but only {actual_steps} available in historical data. "
                f"Using all available data."
            )

        # Extract spot prices
        spot_prices = torch.tensor(
            data["last_price"].values, dtype=self.dtype, device=self.device
        ).unsqueeze(
            0
        )  # Shape: (1, n_steps)

        # Replicate paths if needed
        if n_paths > 1:
            spot_prices = spot_prices.repeat(n_paths, 1)

        self.register_buffer("spot", spot_prices)

        # Load bid/ask spreads
        if "bid_price" in data.columns and "ask_price" in data.columns:
            bid_prices = torch.tensor(
                data["bid_price"].values, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            ask_prices = torch.tensor(
                data["ask_price"].values, dtype=self.dtype, device=self.device
            ).unsqueeze(0)

            if n_paths > 1:
                bid_prices = bid_prices.repeat(n_paths, 1)
                ask_prices = ask_prices.repeat(n_paths, 1)
        else:
            # Estimate bid/ask from spot with typical spread
            spread = 0.0002  # 2 basis points
            bid_prices = spot_prices * (1 - spread / 2)
            ask_prices = spot_prices * (1 + spread / 2)

        self.register_buffer("bid", bid_prices)
        self.register_buffer("ask", ask_prices)
        self.register_buffer("mid", (bid_prices + ask_prices) / 2)

        # Load funding rates
        if "funding_rate" in data.columns:
            funding_rates = torch.tensor(
                data["funding_rate"].values, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            if n_paths > 1:
                funding_rates = funding_rates.repeat(n_paths, 1)
        else:
            # Default to zero funding if not available
            funding_rates = torch.zeros_like(spot_prices)

        self.register_buffer("_funding_rate", funding_rates)

        # Load index price
        if "index_price" in data.columns:
            index_prices = torch.tensor(
                data["index_price"].values, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            if n_paths > 1:
                index_prices = index_prices.repeat(n_paths, 1)
        else:
            # Use spot price as index if not available
            index_prices = spot_prices.clone()

        self.register_buffer("index_price", index_prices)

    @property
    def volatility(self) -> Tensor:
        if not hasattr(self, "spot"):
            raise ValueError("No data loaded. Call simulate() first.")

        # Use mixin for rolling window or constant vol
        if self.constant_volatility is not None or self.volatility_window > 0:
            return self.calculate_volatility()

        # Fallback: expanding window (backward compatibility when volatility_window=0)
        spot = self.get_buffer("spot")
        returns = torch.log(spot[:, 1:] / spot[:, :-1])

        # Annualized volatility (use actual dt, not hardcoded 5-min assumption)
        periods_per_year = (
            1.0 / self.dt if self.dt > 0 else 365 * 24 * 12
        )  # fallback to 5-min

        if returns.shape[1] > 0:
            # Use expanding window volatility
            vol_list = []
            for i in range(returns.shape[1]):
                if i == 0:
                    vol = torch.full(
                        (returns.shape[0], 1), 0.8, dtype=self.dtype, device=self.device
                    )
                else:
                    hist_returns = returns[:, : i + 1]
                    vol = torch.std(hist_returns, dim=1, keepdim=True) * (
                        periods_per_year**0.5
                    )
                vol_list.append(vol)

            initial_vol = torch.full(
                (returns.shape[0], 1), 0.8, dtype=self.dtype, device=self.device
            )
            vol_list.insert(0, initial_vol)

            volatility = torch.cat(vol_list, dim=1)
        else:
            volatility = torch.full_like(spot, 0.8)

        return volatility

    @property
    def variance(self) -> Tensor:
        return self.volatility**2

    @staticmethod
    def validate_bootstrap_data_sufficiency(
        n_paths: int,
        n_steps: int,
        available_records: int,
        dt: float,
    ) -> Tuple[bool, Optional[str], int]:
        max_start = available_records - n_steps
        unique_windows = max_start + 1 if max_start >= 0 else 0
        recommended_windows = max(100, n_paths // 10)
        recommended_records = n_steps + recommended_windows

        # Check if we can even run bootstrap
        if max_start < 0:
            error_msg = (
                f"Insufficient data: need {n_steps} steps but only have {available_records} records.\n"
                f"Minimum required: {n_steps} records\n"
                f"Recommended: {recommended_records} records"
            )
            return False, error_msg, recommended_records

        # Check if we have zero variance (only 1 window)
        if max_start == 0:
            warning_msg = (
                f"\n{'='*80}\n"
                f"⚠️  BOOTSTRAP VARIANCE WARNING: Only 1 possible window!\n"
                f"{'='*80}\n"
                f"All {n_paths} paths will be IDENTICAL (zero variance in results).\n\n"
                f"Current data: {available_records} records\n"
                f"Backtest needs: {n_steps} steps\n"
                f"Available windows: {unique_windows}\n\n"
                f"SOLUTION: Download more historical data!\n"
                f"  Recommended: {recommended_records} records\n"
                f"  This enables: {recommended_windows} unique bootstrap windows\n"
                f"  Extra days needed: ~{(recommended_windows * dt * 365):.0f} days\n"
                f"{'='*80}\n"
            )
            return False, warning_msg, recommended_records

        # Check if we have limited variance (few windows)
        if unique_windows < n_paths // 10:
            warning_msg = (
                f"\n{'='*80}\n"
                f"⚠️  BOOTSTRAP VARIANCE WARNING: Limited sampling diversity\n"
                f"{'='*80}\n"
                f"Only {unique_windows} unique windows available for {n_paths} paths.\n"
                f"Many paths will be duplicates (reduces statistical power).\n\n"
                f"For better variance, download more historical data:\n"
                f"  Current: {available_records} records ({unique_windows} windows)\n"
                f"  Recommended: {recommended_records} records ({recommended_windows} windows)\n"
                f"{'='*80}\n"
            )
            return False, warning_msg, recommended_records

        # Sufficient data
        return True, None, recommended_records

    def simulate_bootstrap(
        self,
        n_paths: int,
        time_horizon: float,
        window_size: Optional[int] = None,
        target_initial_spot: Optional[float] = None,
        max_date: Optional[str] = None,
        store_scale_factors: bool = False,
    ) -> None:
        import math
        import random
        import logging

        logger = logging.getLogger(__name__)

        # Load all available data
        # Prefer full data for bootstrap variance, fall back to filtered data or reload
        if (
            hasattr(self.data_loader, "perpetual_data_full")
            and self.data_loader.perpetual_data_full is not None
            and not self.data_loader.perpetual_data_full.empty
        ):
            # Use full data (includes all history for bootstrap variance)
            perpetual_df = self.data_loader.perpetual_data_full.copy()
        elif (
            self.data_loader.perpetual_data is not None
            and not self.data_loader.perpetual_data.empty
        ):
            # Fall back to filtered data if full not available
            perpetual_df = self.data_loader.perpetual_data.copy()
        else:
            # Last resort: reload from disk
            perpetual_df = self.data_loader.load_perpetual_data()

        if perpetual_df is None or perpetual_df.empty:
            raise ValueError("No perpetual data available")

        # Apply no look-ahead filter
        if max_date is not None:
            max_timestamp = pd.to_datetime(max_date, utc=True)
            if "timestamp" in perpetual_df.columns:
                perpetual_df = perpetual_df[perpetual_df["timestamp"] < max_timestamp]
                logger.info(
                    f"Filtered data to before {max_date}: {len(perpetual_df)} records"
                )

        # Calculate n_steps (single source of truth - match PFHedge convention)
        n_steps = int(time_horizon / self.dt) + 1

        # Validate sufficient data
        total_available = len(perpetual_df)
        if window_size is None:
            window_size = total_available
        else:
            window_size = min(window_size, total_available)

        if window_size < n_steps:
            raise ValueError(
                f"Insufficient historical data for bootstrap:\n"
                f"  Need: {n_steps} steps ({time_horizon*365:.1f} days at dt={self.dt*365:.2f} days)\n"
                f"  Available: {window_size} records\n"
                f"  Suggestion: Reduce maturity_days or fetch more historical data"
            )

        # Calculate max unique paths first
        max_start = window_size - n_steps
        max_unique_paths = max_start + 1

        # Auto-optimize path number if not specified or exceeds maximum
        requested_paths = n_paths
        if n_paths is None or n_paths <= 0:
            n_paths = max_unique_paths
            logger.info(
                f"ℹ️  No path number specified, using max unique paths: {n_paths:,}"
            )
        elif n_paths > max_unique_paths:
            logger.warning(
                f"⚠️  Requested {requested_paths:,} paths exceeds maximum unique paths {max_unique_paths:,}. "
                f"Using {max_unique_paths:,} paths instead."
            )
            n_paths = max_unique_paths

        # Validate data sufficiency and warn if needed
        is_sufficient, warning_msg, _ = self.validate_bootstrap_data_sufficiency(
            n_paths=n_paths,
            n_steps=n_steps,
            available_records=window_size,
            dt=self.dt,
        )

        if warning_msg:
            logger.warning(warning_msg)

        # Initialize lists
        path_list = []
        bid_list = []
        ask_list = []
        mid_list = []
        funding_list = []
        index_list = []
        scale_factors = []
        sampled_windows = []  # Track which windows were sampled for debugging

        # Helper to extract and rescale column
        def get_rescaled_column(
            window_data, col_name: str, rescale_factor: float, default_val=None
        ):
            if col_name in window_data.columns:
                vals = window_data[col_name].values
                return torch.as_tensor(
                    vals * rescale_factor, dtype=self.dtype, device=self.device
                )
            elif default_val is not None:
                return default_val
            else:
                return None

        # Bootstrap sampling
        for path_idx in range(n_paths):
            # Random starting point
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + n_steps
            sampled_windows.append(start_idx)

            # Extract window
            window_data = perpetual_df.iloc[start_idx:end_idx].copy()

            # Handle NaNs/gaps
            if window_data["last_price"].isna().any():
                logger.warning(
                    f"Path {path_idx}: Found NaN in window at index {start_idx}, forward-filling"
                )
                window_data = window_data.ffill().bfill()

            # Extract spot price (required)
            spot_raw = window_data["last_price"].values

            # Calculate rescale factor
            if target_initial_spot is not None:
                rescale_factor = target_initial_spot / spot_raw[0]
                scale_factors.append(rescale_factor)
            else:
                rescale_factor = 1.0
                scale_factors.append(1.0)

            # Rescale spot price
            spot = get_rescaled_column(window_data, "last_price", rescale_factor)
            if spot is None:
                raise ValueError(f"Required column 'last_price' not found in data")
            path_list.append(spot)

            # Bid/ask prices (try multiple column names)
            bid = get_rescaled_column(
                window_data, "best_bid_price", rescale_factor
            ) or get_rescaled_column(window_data, "bid_price", rescale_factor)
            ask = get_rescaled_column(
                window_data, "best_ask_price", rescale_factor
            ) or get_rescaled_column(window_data, "ask_price", rescale_factor)

            if bid is None or ask is None:
                # Derive from spot with typical spread
                spread = 0.0002  # 2 bps
                bid = spot * (1 - spread / 2)
                ask = spot * (1 + spread / 2)

            bid_list.append(bid)
            ask_list.append(ask)
            mid_list.append((bid + ask) / 2)

            # Index price (rescale if present)
            index = get_rescaled_column(
                window_data, "index_price", rescale_factor, default_val=spot.clone()
            )
            index_list.append(index)

            # Funding rate (DO NOT rescale - it's a rate, not a price)
            if "funding_rate" in window_data.columns:
                funding_vals = window_data["funding_rate"].values
                funding = torch.as_tensor(
                    funding_vals, dtype=self.dtype, device=self.device
                )
            else:
                funding = torch.zeros_like(spot)
            funding_list.append(funding)

        # Stack all paths
        self.register_buffer("spot", torch.stack(path_list))
        self.register_buffer("bid", torch.stack(bid_list))
        self.register_buffer("ask", torch.stack(ask_list))
        self.register_buffer("mid", torch.stack(mid_list))
        self.register_buffer("_funding_rate", torch.stack(funding_list))
        self.register_buffer("index_price", torch.stack(index_list))

        # Store scale factors for auditability
        if store_scale_factors or target_initial_spot is not None:
            self.register_buffer(
                "_bootstrap_scale_factors",
                torch.as_tensor(scale_factors, dtype=self.dtype, device=self.device),
            )

        # Logging
        unique_windows = len(set(sampled_windows))
        window_stats = f"windows: {unique_windows} unique (range: {min(sampled_windows)}-{max(sampled_windows)})"

        if target_initial_spot is not None:
            logger.info(
                f"Bootstrap: Rescaled {n_paths} paths to initial_spot=${target_initial_spot:,.2f} "
                f"(scale: {min(scale_factors):.3f}-{max(scale_factors):.3f}), {window_stats}"
            )
        else:
            logger.info(
                f"Bootstrap: Sampled {n_paths} paths (no rescaling), {window_stats}"
            )

        # Store window indices for debugging (optional debug logging)
        logger.debug(f"Bootstrap window indices: {sampled_windows}")

    def __repr__(self) -> str:
        params = [
            f"cost={self.cost}",
            f"dt={self.dt}",
            f"leverage={self.leverage}",
            "data_loader=...",
        ]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinPerpetualHistorical({', '.join(params)})"
