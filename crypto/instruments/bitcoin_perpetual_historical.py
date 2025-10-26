"""
Bitcoin perpetual with historical data replay for backtesting.
"""
from typing import Optional, Tuple
import pandas as pd
import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar

from .bitcoin_perpetual_base import BitcoinPerpetualBase


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


class BitcoinPerpetualHistorical(BitcoinPerpetualBase):
    """Bitcoin perpetual using historical data for backtesting.

    This implementation loads real historical data for backtesting strategies.
    Unlike BitcoinPerpetualBrownian which generates synthetic paths for training,
    this uses actual market data to test performance on historical scenarios.

    Args:
        data_loader (CryptoDataLoader): Data loader for historical data.
        cost (float, default=0.0006): Transaction cost rate.
        dt (float, default=1/24/12): Time step (5 minutes).
        leverage (float, default=20.0): Maximum leverage.
        dtype (torch.dtype, optional): Tensor dtype.
        device (torch.device, optional): Tensor device.

    Examples:
        >>> from crypto.data.loader import CryptoDataLoader
        >>> loader = CryptoDataLoader("sample_data")
        >>> btc = BitcoinPerpetualHistorical(data_loader=loader)
        >>> btc.simulate(n_paths=1, time_horizon=30/365)
        >>> print(btc.spot.shape)
        torch.Size([1, 8640])  # 1 path, 30 days of 5-min data
        >>>
        >>> # Backtesting a strategy
        >>> from pfhedge.instruments import EuropeanOption
        >>> option = EuropeanOption(btc, strike=50000, maturity=30/365)
        >>> # Now can backtest hedging strategies on real data
    """

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
    ) -> None:
        """Initialize historical Bitcoin perpetual.

        Args:
            data_loader: Data loader for historical data
            cost: Transaction cost rate
            dt: Time step in years
            leverage: Maximum leverage
            dtype: Tensor dtype
            device: Tensor device
            constant_volatility: If provided, use this constant volatility instead of
                calculating from returns. This should match the training volatility to
                avoid train/test distribution mismatch.
        """
        super().__init__(
            cost=cost, dt=dt, leverage=leverage, dtype=dtype, device=device
        )

        if data_loader is None:
            raise ValueError("data_loader is required for BitcoinPerpetualHistorical")

        self.data_loader = data_loader
        self.constant_volatility = (
            constant_volatility  # NEW: Store for use in volatility property
        )

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Load historical data for backtesting.

        For backtesting, we typically use n_paths=1 to test on actual history.
        If n_paths > 1 is requested, we replicate the same historical path
        multiple times (useful for Monte Carlo with transaction costs).

        Args:
            n_paths: Number of paths (typically 1 for backtesting)
            time_horizon: Time period to load
            init_state: Not used for historical data (uses actual prices)

        Examples:
            >>> btc = BitcoinPerpetualHistorical(data_loader=loader)
            >>> btc.simulate(n_paths=1, time_horizon=30/365)
            >>> # Now have actual historical data for backtesting
            >>>
            >>> # Can also replicate for Monte Carlo with costs
            >>> btc.simulate(n_paths=100, time_horizon=30/365)
            >>> # 100 identical paths, useful for averaging over random trades
        """
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
        """Returns volatility for the instrument.

        If constant_volatility was provided at initialization, returns that constant value.
        Otherwise, calculates rolling volatility from actual price movements.

        Using constant volatility matching the training volatility prevents train/test
        distribution mismatch when the model was trained on simulated GBM data.
        """
        if not hasattr(self, "spot"):
            raise ValueError("No data loaded. Call simulate() first.")

        spot = self.get_buffer("spot")

        # Use constant volatility if provided (for train/test consistency)
        if self.constant_volatility is not None:
            return torch.full_like(spot, self.constant_volatility)

        # Otherwise calculate from returns
        returns = torch.log(spot[:, 1:] / spot[:, :-1])

        # Annualized volatility (use actual dt, not hardcoded 5-min assumption)
        # periods_per_year = 1 / dt, where dt is in years
        periods_per_year = (
            1.0 / self.dt if self.dt > 0 else 365 * 24 * 12
        )  # fallback to 5-min

        if returns.shape[1] > 0:
            # Use expanding window volatility
            vol_list = []
            for i in range(returns.shape[1]):
                if i == 0:
                    # Use a default volatility for the first period
                    vol = torch.full(
                        (returns.shape[0], 1), 0.8, dtype=self.dtype, device=self.device
                    )
                else:
                    # Calculate volatility up to current point
                    hist_returns = returns[:, : i + 1]
                    vol = torch.std(hist_returns, dim=1, keepdim=True) * (
                        periods_per_year ** 0.5
                    )
                vol_list.append(vol)

            # Add initial volatility for time 0
            initial_vol = torch.full(
                (returns.shape[0], 1), 0.8, dtype=self.dtype, device=self.device
            )
            vol_list.insert(0, initial_vol)

            volatility = torch.cat(vol_list, dim=1)
        else:
            # No returns, use default
            volatility = torch.full_like(spot, 0.8)

        return volatility

    @property
    def variance(self) -> Tensor:
        """Returns historical realized variance."""
        return self.volatility ** 2

    @staticmethod
    def validate_bootstrap_data_sufficiency(
        n_paths: int,
        n_steps: int,
        available_records: int,
        dt: float,
    ) -> Tuple[bool, Optional[str], int]:
        """Validate whether historical data is sufficient for meaningful bootstrap variance.

        Args:
            n_paths: Number of bootstrap paths to generate.
            n_steps: Number of time steps required per path.
            available_records: Number of historical records available.
            dt: Time step size in years (e.g., 8 hours = 8/24/365).

        Returns:
            Tuple of (is_sufficient, warning_message, recommended_records):
                - is_sufficient: True if variance will be meaningful, False otherwise.
                - warning_message: Detailed warning if insufficient, None otherwise.
                - recommended_records: Recommended number of records for good variance.

        Examples:
            >>> is_ok, msg, rec = BitcoinPerpetualHistorical.validate_bootstrap_data_sufficiency(
            ...     n_paths=100, n_steps=42, available_records=50, dt=8/24/365
            ... )
            >>> print(f"Sufficient: {is_ok}, Recommended: {rec}")
            Sufficient: False, Recommended: 52
        """
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
        """Bootstrap simulation with optional spot rescaling.

        This method creates multiple paths by randomly selecting different
        starting points in the historical data. Optionally rescales prices
        to preserve moneyness across all paths.

        Args:
            n_paths: Number of bootstrap paths to generate
            time_horizon: Time period for each path (in years)
            window_size: Size of historical window to sample from (if None, uses all)
            target_initial_spot: If provided, rescale all paths to this initial spot
            max_date: If provided, only sample from data < max_date (no look-ahead)
            store_scale_factors: Whether to store scale factors for auditability

        Examples:
            >>> btc = BitcoinPerpetualHistorical(data_loader=loader)
            >>> # Basic bootstrap (no rescaling)
            >>> btc.simulate_bootstrap(n_paths=1000, time_horizon=5/365)
            >>>
            >>> # Moneyness-preserving bootstrap
            >>> btc.simulate_bootstrap(
            ...     n_paths=1000,
            ...     time_horizon=5/365,
            ...     target_initial_spot=108000,  # Rescale all paths to start at $108k
            ...     max_date="2024-10-15"  # No look-ahead
            ... )
        """
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

        # Validate data sufficiency and warn if needed
        is_sufficient, warning_msg, _ = self.validate_bootstrap_data_sufficiency(
            n_paths=n_paths,
            n_steps=n_steps,
            available_records=window_size,
            dt=self.dt,
        )

        if warning_msg:
            logger.warning(warning_msg)

        max_start = window_size - n_steps

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
            """Extract column and apply rescale factor if it's a price column."""
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
        """String representation."""
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
