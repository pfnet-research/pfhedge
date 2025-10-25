"""Backtesting framework for deep hedging strategies."""

from typing import Optional, TYPE_CHECKING
import os
import pickle
import torch
from torch import Tensor

from .config import BacktestConfig

if TYPE_CHECKING:
    from pfhedge.nn import Hedger
    from crypto.data.loader import CryptoDataLoader
    from crypto.instruments import BitcoinEuropeanOption


class Backtester:
    """Backtest deep hedging strategies on historical data.

    This class orchestrates the backtesting process:
    1. Load pre-trained model
    2. Load historical data
    3. Create bootstrap paths from historical data
    4. Run deep hedge and Black-Scholes strategies
    5. Calculate performance metrics
    6. Generate reports and visualizations

    Args:
        config: Backtest configuration

    Examples:
        >>> from crypto.backtest.config import BacktestConfig
        >>> config = BacktestConfig(
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     strike=50000,
        ...     maturity_days=14,
        ...     model_path="models/deep_hedger.pth"
        ... )
        >>> backtester = Backtester(config)
        >>> results = backtester.run()
    """

    def __init__(self, config: BacktestConfig):
        """Initialize backtester with configuration.

        Args:
            config: Backtest configuration
        """
        self.config = config

        # Placeholders for loaded components (will be set by methods)
        self.model = None
        self.data_loader = None
        self.option = None

        # Placeholders for strategy results (set during run)
        self.deep_positions = None
        self.bs_positions = None

    def load_model(self, device: Optional[str] = None) -> "Hedger":
        """Load pre-trained model from checkpoint.

        The checkpoint should contain:
        - 'model_state_dict': The model's state dictionary
        - 'model_config': Dict with n_layers, n_units, criterion/risk_measure, risk_param
        - 'features' (optional): Falls back to DEFAULT_FEATURES if missing

        Args:
            device: Target device ('cpu', 'cuda', etc.). If None, uses 'cpu'.

        Returns:
            Loaded Hedger model

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint is missing required keys
            RuntimeError: If state dict doesn't match model architecture

        Examples:
            >>> backtester = Backtester(config)
            >>> model = backtester.load_model()  # Load to CPU
            >>> model = backtester.load_model(device='cuda')  # Load to GPU
        """
        # Import here to avoid circular dependency
        from crypto.strategies.deep_hedge_utils import (
            create_deep_hedger,
            DEFAULT_FEATURES,
        )
        from pfhedge.nn import Hedger

        model_path = self.config.model_path

        # Determine device
        if device is None:
            device = "cpu"

        # Check if file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Load checkpoint with safety measures
        print(f"Loading model from {model_path}...")

        # Try safer loading first (PyTorch >= 2.4)
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError):
            # Fall back to unsafe loading for older PyTorch or complex objects
            # (catches UnpicklingError when checkpoint contains non-allowlisted types)
            checkpoint = torch.load(model_path, map_location=device)

        # Validate checkpoint structure
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict'")
        if "model_config" not in checkpoint:
            raise KeyError("Checkpoint missing 'model_config'")

        # Extract model configuration
        model_config = checkpoint["model_config"]

        # Check required keys (with backward compatibility)
        required_keys = ["n_layers", "n_units", "risk_param"]
        for key in required_keys:
            if key not in model_config:
                raise KeyError(f"Model config missing required key: '{key}'")

        # Handle backward compatibility for criterion/risk_measure
        if "criterion" in model_config:
            criterion = model_config["criterion"]
        elif "risk_measure" in model_config:
            criterion = model_config["risk_measure"]
        else:
            raise KeyError("Model config missing 'criterion' or 'risk_measure'")

        # Handle features with default fallback
        if "features" in model_config:
            features = model_config["features"]
        else:
            features = DEFAULT_FEATURES
            print(
                f"⚠️  Warning: 'features' not found in checkpoint, using DEFAULT_FEATURES"
            )

        # Create model with same architecture
        model = create_deep_hedger(
            n_layers=model_config["n_layers"],
            n_units=model_config["n_units"],
            risk_measure=criterion,
            risk_param=model_config["risk_param"],
            features=features,
        )

        # Load trained weights with strict checking
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load model weights. State dict mismatch: {e}\n"
                f"This usually means the checkpoint was saved with a different model architecture."
            )

        # Set to evaluation mode
        model.eval()

        # Store and return
        self.model = model
        print(f"✅ Model loaded successfully")
        print(f"   Device: {device}")
        print(
            f"   Architecture: {model_config['n_layers']} layers × {model_config['n_units']} units"
        )
        print(f"   Features: {features}")
        print(f"   Criterion: {criterion} (param={model_config['risk_param']})")

        return model

    def _normalize_timestamps_to_utc(self, df, timestamp_col="timestamp"):
        """Normalize timestamp column to UTC timezone.

        Handles both timezone-naive and timezone-aware timestamps:
        - If naive: localize to UTC
        - If aware but not UTC: convert to UTC
        - If already UTC: no change

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column (default: "timestamp")

        Returns:
            DataFrame with timestamps normalized to UTC

        Note:
            This modifies the DataFrame in-place for efficiency.
        """
        if timestamp_col not in df.columns:
            return df

        if df[timestamp_col].dt.tz is None:
            # Timezone-naive: localize to UTC
            df[timestamp_col] = df[timestamp_col].dt.tz_localize("UTC")
        elif str(df[timestamp_col].dt.tz) != "UTC":
            # Timezone-aware but not UTC: convert to UTC
            df[timestamp_col] = df[timestamp_col].dt.tz_convert("UTC")
        # else: already UTC, no change needed

        return df

    def load_data(self) -> "CryptoDataLoader":
        """Load historical data for backtesting.

        Loads perpetual and options data from parquet files using CryptoDataLoader,
        resamples to the configured frequency (dt_hours), and filters by date range.

        All timestamps are normalized to UTC for consistent comparison.

        Returns:
            CryptoDataLoader with historical data loaded and prepared

        Raises:
            FileNotFoundError: If data directory doesn't exist or no data files found
            ValueError: If data loading fails, data is empty, or date filtering results in no data

        Examples:
            >>> backtester = Backtester(config)
            >>> loader = backtester.load_data()
            >>> print(loader.summary())  # Verify data loaded
        """
        # Import CryptoDataLoader and pandas
        from crypto.data.loader import CryptoDataLoader
        import pandas as pd
        from datetime import datetime

        data_dir = self.config.data_dir

        # Trust the path from config (already resolved by YAML loader if loaded from file)
        # Just validate that it exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Tip: If using YAML config, relative paths are resolved relative to the config file.\n"
                f"      If creating config programmatically, use absolute paths or resolve manually."
            )

        print(f"Loading historical data from {data_dir}...")

        # Create data loader
        loader = CryptoDataLoader(data_dir)

        # Load perpetual data (required for bootstrapping)
        try:
            perpetual_df = loader.load_perpetual_data()
            print(f"✅ Loaded {len(perpetual_df)} raw perpetual records")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No perpetual data found in {data_dir}. "
                f"Expected files matching '*perpetual*.parquet'. Error: {e}"
            )

        if perpetual_df.empty:
            raise ValueError("Perpetual data is empty")

        # Derive resampling frequency from dt_hours
        dt_hours = self.config.dt_hours
        if dt_hours == int(dt_hours):
            # Integer hours
            frequency = f"{int(dt_hours)}H"
        else:
            # Convert to minutes
            dt_minutes = int(round(dt_hours * 60))
            frequency = f"{dt_minutes}T"

        print(f"Resampling to {frequency} frequency...")

        # Resample perpetual data using get_price_series
        try:
            resampled_df = loader.get_price_series(frequency=frequency)
            print(
                f"✅ Resampled to {len(resampled_df)} records at {frequency} intervals"
            )
        except Exception as e:
            raise ValueError(f"Failed to resample data at frequency '{frequency}': {e}")

        # Normalize perpetual timestamps to UTC for consistent comparison
        resampled_df = self._normalize_timestamps_to_utc(resampled_df)

        # Parse date range
        try:
            start_date = pd.to_datetime(self.config.start_date)
            end_date = pd.to_datetime(self.config.end_date)
        except Exception as e:
            raise ValueError(f"Failed to parse dates: {e}")

        # Ensure comparison dates are UTC-aware to match normalized timestamps
        if start_date.tz is None:
            start_date = start_date.tz_localize("UTC")
        if end_date.tz is None:
            end_date = end_date.tz_localize("UTC")

        # Filter by date range
        print(
            f"Filtering data from {self.config.start_date} to {self.config.end_date}..."
        )
        mask = (resampled_df["timestamp"] >= start_date) & (
            resampled_df["timestamp"] <= end_date
        )
        filtered_df = resampled_df[mask].reset_index(drop=True)

        if filtered_df.empty:
            raise ValueError(
                f"No data found in date range [{self.config.start_date}, {self.config.end_date}]. "
                f"Available data range: [{resampled_df['timestamp'].min()}, {resampled_df['timestamp'].max()}]"
            )

        print(f"✅ Filtered to {len(filtered_df)} records in date range")

        # IMPORTANT: Store BOTH filtered and full data
        # - perpetual_data: filtered data (for backward compatibility, tests, validation)
        # - perpetual_data_full: full resampled data (for bootstrap variance)
        # Bootstrap needs access to ALL historical data, not just the filtered backtest range
        loader.perpetual_data = filtered_df  # Filtered data for backward compatibility
        loader.perpetual_data_full = resampled_df  # Full data for bootstrap variance

        # Load options data (optional, for real option price comparison)
        try:
            options_df = loader.load_options_data()

            # Normalize options timestamps to UTC (same as perpetual data)
            if not options_df.empty and "timestamp" in options_df.columns:
                options_df = self._normalize_timestamps_to_utc(options_df)

                # Filter options by date range (dates already UTC-aware)
                opts_mask = (options_df["timestamp"] >= start_date) & (
                    options_df["timestamp"] <= end_date
                )
                options_df = options_df[opts_mask].reset_index(drop=True)
                loader.options_data = options_df

            print(f"✅ Loaded {len(options_df)} options records in date range")
        except FileNotFoundError:
            print(f"⚠️  No options data found (this is okay for basic backtesting)")
            options_df = None

        # Load funding data (optional, but recommended for realistic backtesting)
        try:
            funding_df = loader.load_funding_data()

            if not funding_df.empty and "timestamp" in funding_df.columns:
                funding_df = self._normalize_timestamps_to_utc(funding_df)

                # Don't filter funding by date - merge with full data for bootstrap
                # Filter just for display/validation
                funding_mask = (funding_df["timestamp"] >= start_date) & (
                    funding_df["timestamp"] <= end_date
                )
                funding_in_range = funding_df[funding_mask]

                print(
                    f"✅ Loaded {len(funding_in_range)} funding rate records in date range"
                )

                # Merge funding rates with BOTH filtered and full perpetual data
                if not funding_df.empty:
                    # Merge with FULL resampled data (for bootstrap variance)
                    merged_full = resampled_df.merge(
                        funding_df[["timestamp", "interest_8h"]],
                        on="timestamp",
                        how="left",
                    )
                    # Rename for clarity
                    if "interest_8h" in merged_full.columns:
                        merged_full["funding_rate"] = merged_full["interest_8h"]
                    # Forward and backward fill missing funding rates
                    if "funding_rate" in merged_full.columns:
                        merged_full["funding_rate"] = (
                            merged_full["funding_rate"].ffill().bfill()
                        )

                    # Merge with filtered data (for backward compatibility)
                    merged_filtered = filtered_df.merge(
                        funding_df[["timestamp", "interest_8h"]],
                        on="timestamp",
                        how="left",
                    )
                    if "interest_8h" in merged_filtered.columns:
                        merged_filtered["funding_rate"] = merged_filtered["interest_8h"]
                    if "funding_rate" in merged_filtered.columns:
                        merged_filtered["funding_rate"] = (
                            merged_filtered["funding_rate"].ffill().bfill()
                        )

                    # Update loader with BOTH versions
                    loader.perpetual_data = (
                        merged_filtered  # Filtered (for tests/validation)
                    )
                    loader.perpetual_data_full = (
                        merged_full  # Full (for bootstrap variance)
                    )
        except FileNotFoundError:
            print(
                f"⚠️  No funding data found (this is okay, but funding costs won't be applied)"
            )

        # Print summary statistics to verify real market data
        summary = loader.summary()
        print("\n" + "=" * 60)
        print("DATA SUMMARY (After Filtering & Resampling)")
        print("=" * 60)

        if "perpetual" in summary:
            perp = summary["perpetual"]
            print(f"\nPerpetual data:")
            print(f"  Records: {perp['records']:,}")
            print(f"  Frequency: {frequency}")
            print(f"  Date range: {perp['date_range'][0]} to {perp['date_range'][1]}")
            print(
                f"  Price range: ${perp['price_range'][0]:.2f} - ${perp['price_range'][1]:.2f}"
            )
            if perp.get("avg_spread_pct"):
                print(f"  Avg spread: {perp['avg_spread_pct']*100:.4f}%")

        if "options" in summary:
            opts = summary["options"]
            print(f"\nOptions data:")
            print(f"  Records: {opts['records']:,}")
            print(f"  Unique strikes: {opts['unique_strikes']}")
            print(f"  Calls: {opts['call_count']:,}")
            print(f"  Puts: {opts['put_count']:,}")
            if opts.get("avg_iv"):
                print(f"  Avg IV: {opts['avg_iv']:.2%}")

        print("=" * 60 + "\n")

        # Store and return
        self.data_loader = loader
        print("✅ Data loading complete\n")

        return loader

    def create_bootstrap_option(
        self, data_loader: Optional["CryptoDataLoader"] = None
    ) -> "BitcoinEuropeanOption":
        """Create option with bootstrap paths from historical data.

        Uses BitcoinPerpetualHistorical to generate multiple bootstrap paths
        from the loaded historical data, then creates a European option on top.

        Args:
            data_loader: CryptoDataLoader with historical data.
                        If None, uses self.data_loader.

        Returns:
            BitcoinEuropeanOption with bootstrap paths from historical data

        Raises:
            ValueError: If data_loader is None and no data has been loaded
            ValueError: If required config parameters are missing

        Examples:
            >>> backtester = Backtester(config)
            >>> backtester.load_model()
            >>> loader = backtester.load_data()
            >>> option = backtester.create_bootstrap_option(loader)
            >>> print(option.summary())
        """
        # Import necessary classes
        import logging
        import numpy as np
        from crypto.instruments import BitcoinPerpetualHistorical, BitcoinEuropeanOption

        logger = logging.getLogger(__name__)

        # Use provided data_loader or fall back to self.data_loader
        if data_loader is None:
            data_loader = self.data_loader

        if data_loader is None:
            raise ValueError(
                "No data loaded. Call load_data() first or provide a data_loader."
            )

        # Validate that data exists
        if data_loader.perpetual_data is None or data_loader.perpetual_data.empty:
            raise ValueError("Data loader has no perpetual data")

        logger.info("Creating bootstrap option from historical data...")

        # Create BitcoinPerpetualHistorical with loaded data
        underlier = BitcoinPerpetualHistorical(
            data_loader=data_loader,
            cost=self.config.transaction_cost,
            dt=self.config.dt,
            dtype=torch.float32,
            device="cpu",
        )

        # Calculate time horizon from maturity_days
        time_horizon = self.config.maturity_days / 365.0

        # Determine bootstrap parameters based on mode
        bootstrap_mode = self.config.bootstrap_mode

        if bootstrap_mode == "normalize_spot":
            # Calculate target initial spot
            target_moneyness = self.config.effective_target_moneyness
            target_initial_spot = self.config.strike * target_moneyness

            logger.info(f"Bootstrap mode: normalize_spot")
            logger.info(f"  Target moneyness: {target_moneyness:.4f}")
            logger.info(f"  Target initial spot: ${target_initial_spot:,.2f}")
            logger.info(f"  Strike: ${self.config.strike:,.2f}")
            logger.info(f"  Target log_moneyness: {np.log(target_moneyness):.4f}")

            # Generate bootstrap with rescaling (no look-ahead)
            try:
                underlier.simulate_bootstrap(
                    n_paths=self.config.n_bootstrap_paths,
                    time_horizon=time_horizon,
                    window_size=None,
                    target_initial_spot=target_initial_spot,  # Enable rescaling
                    max_date=None,  # For backtesting, use all available data in the loaded range
                    store_scale_factors=True,
                )
            except Exception as e:
                raise ValueError(f"Failed to generate bootstrap paths: {e}")

            logger.info(f"✅ Generated {underlier.spot.shape[0]} bootstrap paths")
            logger.info(f"   Each path has {underlier.spot.shape[1]} time steps")

            # Verify moneyness consistency
            initial_spots = underlier.spot[:, 0]
            initial_moneyness = initial_spots / self.config.strike
            log_moneyness = torch.log(initial_moneyness)

            # Check consistency
            std_log_moneyness = log_moneyness.std().item()
            mean_log_moneyness = log_moneyness.mean().item()
            target_log_moneyness = np.log(target_moneyness)

            if std_log_moneyness > 1e-6:
                logger.warning(
                    f"Initial log_moneyness varies across paths (std={std_log_moneyness:.6f})"
                )

            if abs(mean_log_moneyness - target_log_moneyness) > 1e-6:
                logger.warning(
                    f"Mean log_moneyness ({mean_log_moneyness:.4f}) differs from target ({target_log_moneyness:.4f})"
                )

            logger.info(
                f"✅ Moneyness verified: mean={mean_log_moneyness:.4f}, std={std_log_moneyness:.6f}, "
                f"initial_spot: ${initial_spots.mean().item():,.2f} (${initial_spots.min().item():,.2f}-${initial_spots.max().item():,.2f})"
            )

        elif bootstrap_mode == "absolute_strike":
            # Legacy mode: no rescaling
            logger.info(f"Bootstrap mode: absolute_strike (no rescaling)")
            logger.warning(
                "Using absolute_strike mode - paths will have varying moneyness. "
                "Consider bootstrap_mode='normalize_spot' for consistent results."
            )

            try:
                underlier.simulate_bootstrap(
                    n_paths=self.config.n_bootstrap_paths,
                    time_horizon=time_horizon,
                    window_size=None,
                    target_initial_spot=None,  # No rescaling
                    max_date=None,  # For backtesting, use all available data in the loaded range
                    store_scale_factors=False,
                )
            except Exception as e:
                raise ValueError(f"Failed to generate bootstrap paths: {e}")

            logger.info(f"✅ Generated {underlier.spot.shape[0]} bootstrap paths")
            logger.info(f"   Each path has {underlier.spot.shape[1]} time steps")

            # Log moneyness distribution for visibility
            initial_spots = underlier.spot[:, 0]
            initial_moneyness = initial_spots / self.config.strike
            log_moneyness = torch.log(initial_moneyness)

            # Percentiles
            p5 = torch.quantile(log_moneyness, 0.05).item()
            p50 = torch.quantile(log_moneyness, 0.50).item()
            p95 = torch.quantile(log_moneyness, 0.95).item()

            logger.info(
                f"ℹ️  Initial log_moneyness distribution: "
                f"5th={p5:.4f}, median={p50:.4f}, 95th={p95:.4f} "
                f"(range: {np.exp(p5):.3f}x to {np.exp(p95):.3f}x)"
            )

        # Create BitcoinEuropeanOption on top of the underlier
        option = BitcoinEuropeanOption(
            underlier=underlier,
            strike=self.config.strike,
            maturity=time_horizon,
            call=self.config.call,
            cost=0.0,  # Option itself has no transaction cost (only underlier does)
        )

        # Store for later use
        self.option = option

        # Print summary
        summary = option.summary()
        print("\n" + "=" * 60)
        print("BOOTSTRAP OPTION SUMMARY")
        print("=" * 60)
        print(f"\nOption:")
        print(f"  Type: {'Call' if self.config.call else 'Put'}")
        print(f"  Strike: ${self.config.strike:,.2f}")
        print(f"  Maturity: {self.config.maturity_days} days")
        print(f"  Paths: {summary['n_paths']:,}")
        print(f"  Steps per path: {summary['n_steps']}")

        if summary.get("simulated", False):
            print(f"\nMarket Data:")
            print(f"  Initial spot (avg): ${summary['spot_initial']:,.2f}")
            print(
                f"  Final spot (avg): ${summary['spot_final_mean']:,.2f} ± ${summary['spot_final_std']:,.2f}"
            )
            print(f"\nOption Statistics:")
            print(
                f"  Payoff (avg): ${summary['payoff_mean']:,.2f} ± ${summary['payoff_std']:,.2f}"
            )
            print(f"  ITM ratio: {summary['itm_ratio']:.1%}")

        print("=" * 60 + "\n")

        print("✅ Bootstrap option created successfully\n")

        return option

    def run_deep_hedge(self, option=None, model=None) -> Tensor:
        """Run deep hedging strategy on option.

        Uses the pre-trained neural network to compute optimal hedge positions
        at each time step, then calculates the resulting PnL including transaction
        costs and funding costs.

        Args:
            option: Option to hedge. If None, uses self.option.
            model: Pre-trained Hedger model. If None, uses self.model.

        Returns:
            Cumulative PnL tensor, shape (n_paths, n_steps)

        Raises:
            ValueError: If option or model is None and not previously loaded

        Examples:
            >>> backtester = Backtester(config)
            >>> backtester.load_model()
            >>> loader = backtester.load_data()
            >>> option = backtester.create_bootstrap_option(loader)
            >>> deep_pnl = backtester.run_deep_hedge(option, backtester.model)
            >>> print(deep_pnl.shape)  # (n_paths, n_steps)
        """
        # Import utilities
        from crypto.strategies.deep_hedge_utils import compute_funding_cum_cost

        # Use provided option/model or fall back to stored ones
        if option is None:
            option = self.option
        if model is None:
            model = self.model

        if option is None:
            raise ValueError(
                "No option provided. Call create_bootstrap_option() first or provide an option."
            )
        if model is None:
            raise ValueError(
                "No model loaded. Call load_model() first or provide a model."
            )

        print("Running deep hedging strategy...")

        # Set model to eval mode (should already be, but make sure)
        model.eval()

        # Ensure option tensors are on same device as model
        # This handles case where model is on CUDA but option was created on CPU
        model_device = next(model.parameters()).device
        if hasattr(option.underlier, "spot"):
            if option.underlier.spot.device != model_device:
                print(
                    f"   Moving option tensors from {option.underlier.spot.device} to {model_device}"
                )
                option.underlier.to(model_device)

        with torch.no_grad():
            # Compute hedge positions using the neural network
            # model.compute_hedge() returns shape (n_paths, n_instruments, n_steps)
            # We only hedge with one underlier, so squeeze dimension 1 (n_instruments)
            # Using squeeze(1) ensures we preserve (n_paths, n_steps) even when n_paths=1
            hedge_positions = model.compute_hedge(option).squeeze(1)

            # Compute cumulative PnL using the model's built-in method
            # This already includes transaction costs from the underlier
            # model.compute_cum_pl() already returns shape (n_paths, n_steps), no squeezing needed
            cum_pnl = model.compute_cum_pl(option)

            # Add funding costs for perpetual futures
            if hasattr(option.underlier, "funding_rate") and hasattr(
                option.underlier, "funding_payment_times"
            ):
                spots = option.underlier.spot
                funding_rate = option.underlier.funding_rate
                funding_times = option.underlier.funding_payment_times()

                # Calculate cumulative funding costs
                funding_costs = compute_funding_cum_cost(
                    spots=spots,
                    positions=hedge_positions,
                    funding_rate=funding_rate,
                    funding_times=funding_times,
                )

                # Subtract funding costs from PnL (costs reduce profit)
                cum_pnl = cum_pnl - funding_costs

                print(
                    f"   Applied funding costs: ${funding_costs[:, -1].mean().item():.2f} avg per path"
                )

        print(f"✅ Deep hedge strategy computed")
        print(f"   Positions shape: {hedge_positions.shape}")
        print(f"   PnL shape: {cum_pnl.shape}")
        print(
            f"   Final PnL: ${cum_pnl[:, -1].mean().item():.2f} ± ${cum_pnl[:, -1].std().item():.2f}"
        )

        # Store positions for later use in results
        self.deep_positions = hedge_positions

        return cum_pnl

    def run_bs_baseline(self, option=None) -> Tensor:
        """Run Black-Scholes delta hedge baseline.

        Uses Black-Scholes delta formula to compute hedge positions at each
        time step, then calculates the resulting PnL including transaction
        costs and funding costs (matching deep hedge calculation).

        Note: Transaction cost (`cost`) comes from the underlier (perpetual
        futures transaction fee), not the option itself. This reflects the
        cost of rebalancing the hedge position in the spot/perpetual market.

        Args:
            option: Option to hedge. If None, uses self.option.

        Returns:
            Cumulative PnL tensor, shape (n_paths, n_steps)

        Raises:
            ValueError: If option is None and not previously created

        Examples:
            >>> backtester = Backtester(config)
            >>> loader = backtester.load_data()
            >>> option = backtester.create_bootstrap_option(loader)
            >>> bs_pnl = backtester.run_bs_baseline(option)
            >>> print(bs_pnl.shape)  # (n_paths, n_steps)
        """
        # Import utilities
        from crypto.strategies.deep_hedge_utils import calculate_bs_hedge_pnl

        # Use provided option or fall back to stored one
        if option is None:
            option = self.option

        if option is None:
            raise ValueError(
                "No option provided. Call create_bootstrap_option() first or provide an option."
            )

        print("Running Black-Scholes delta hedge baseline...")

        # Calculate Black-Scholes delta positions
        # Shape: (n_paths, n_steps)
        bs_delta = option.black_scholes_delta()

        # Get spot prices
        spots = option.underlier.spot

        # Get option payoffs
        payoffs = option.payoff()

        # Get transaction cost from underlier
        cost = option.underlier.cost

        # Get funding rate and funding times if available
        funding_rate = None
        funding_times = None

        if hasattr(option.underlier, "funding_rate") and hasattr(
            option.underlier, "funding_payment_times"
        ):
            funding_rate = option.underlier.funding_rate
            funding_times = option.underlier.funding_payment_times()

        # Calculate cumulative PnL with all costs
        cum_pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=bs_delta,
            payoffs=payoffs,
            cost=cost,
            funding_rate=funding_rate,
            funding_times=funding_times,
        )

        print(f"✅ Black-Scholes baseline computed")
        print(f"   Delta shape: {bs_delta.shape}")
        print(f"   PnL shape: {cum_pnl.shape}")
        print(
            f"   Final PnL: ${cum_pnl[:, -1].mean().item():.2f} ± ${cum_pnl[:, -1].std().item():.2f}"
        )

        # Store positions for later use in results
        self.bs_positions = bs_delta

        return cum_pnl

    def _check_funding_alignment(self) -> None:
        """Check if funding payment times align with resampled grid.

        This is a helper method that warns if funding payments don't align
        with the time grid from bootstrap resampling. Misalignment may result
        in funding costs being applied at slightly different times than intended.

        Note: This is a warning only and doesn't stop execution.
        """
        if self.option is None:
            return

        # Check if underlier has funding payment times
        if not hasattr(self.option.underlier, "funding_payment_times"):
            return

        try:
            funding_times = self.option.underlier.funding_payment_times()
            if funding_times is None or len(funding_times) == 0:
                return

            # Get the time grid from option
            dt = self.config.dt
            n_steps = self.option.underlier.spot.shape[1]
            time_grid = torch.arange(0, n_steps) * dt

            # Check if funding times align with grid (within tolerance)
            tolerance = dt * 0.1  # 10% of time step

            misaligned_times = []
            for t in funding_times:
                # Find closest grid point
                diff = torch.abs(time_grid - float(t))
                closest_idx = torch.argmin(diff)
                closest_time = time_grid[closest_idx]
                if diff[closest_idx] > tolerance:
                    misaligned_times.append(float(t))

            if misaligned_times:
                print("\n" + "⚠️  " * 20)
                print("⚠️  WARNING: Funding Payment Time Alignment Issue")
                print("⚠️  " * 20)
                print(
                    f"\nFound {len(misaligned_times)} funding payment times that don't align"
                )
                print(f"with the resampled time grid (dt={dt:.6f} years).")
                print(f"\nMisaligned times (first 5): {misaligned_times[:5]}")
                print(
                    f"\nThis may cause funding costs to be applied at slightly different"
                )
                print(
                    f"times than intended. Consider adjusting dt_hours to align with funding"
                )
                print(f"payment frequency (typically 8 hours for perpetual futures).")
                print("⚠️  " * 20 + "\n")

        except Exception as e:
            # Don't fail the backtest if alignment check fails
            print(f"⚠️  Note: Could not check funding alignment: {e}")

    def run(self, seed: Optional[int] = None):
        """Run full backtest.

        This orchestrates the entire backtesting process:
        1. Load model
        2. Load data
        3. Create bootstrap option
        4. Run deep hedge strategy
        5. Run BS baseline strategy
        6. Create results object with metrics

        Args:
            seed: Random seed for reproducibility. If None, results may vary
                  between runs due to random bootstrap sampling.

        Returns:
            BacktestResults object with all results and summary statistics

        Raises:
            FileNotFoundError: If model checkpoint or data directory not found
            ValueError: If data loading fails or configuration is invalid
            RuntimeError: If model loading or strategy execution fails

        Examples:
            >>> from crypto.backtest.config import BacktestConfig
            >>> config = BacktestConfig(
            ...     start_date="2024-01-01",
            ...     end_date="2024-01-31",
            ...     strike=50000,
            ...     maturity_days=14,
            ...     model_path="models/deep_hedger.pth"
            ... )
            >>> backtester = Backtester(config)
            >>> results = backtester.run(seed=42)  # Reproducible results
            >>> summary = results.summary()
            >>> print(f"Deep Sharpe: {summary['deep_hedge']['sharpe_ratio']:.3f}")
        """
        # Import BacktestResults
        import logging
        import random
        import numpy as np
        from .results import BacktestResults

        logger = logging.getLogger(__name__)

        # Set random seeds for reproducibility if requested
        if seed is not None:
            # Seed ALL random generators
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Set CUDA seeds for GPU reproducibility
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
                # Enable deterministic mode for cuDNN
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info(f"🔒 Seed={seed} (random, numpy, torch, CUDA)")
            else:
                logger.info(f"🔒 Seed={seed} (random, numpy, torch)")
        else:
            logger.warning("⚠️  No seed set - results may vary between runs")

        print("\n" + "=" * 60)
        print("STARTING BACKTEST")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Date range: {self.config.start_date} to {self.config.end_date}")
        print(
            f"  Option: {'Call' if self.config.call else 'Put'} @ ${self.config.strike:,.2f}"
        )
        print(f"  Maturity: {self.config.maturity_days} days")
        print(f"  Bootstrap paths: {self.config.n_bootstrap_paths}")
        print(f"  Time step: {self.config.dt_hours} hours")
        print(f"  Transaction cost: {self.config.transaction_cost*100:.3f}%")
        print(f"  Model: {self.config.model_path}")
        print(f"  Data directory: {self.config.data_dir}")
        print("=" * 60 + "\n")

        # Wrap execution in try-except to provide helpful error messages
        try:
            # Step 1: Load model
            print("[Step 1/5] Loading model...")
            self.load_model()

            # Step 2: Load data
            print("[Step 2/5] Loading data...")
            self.load_data()

            # Step 3: Create bootstrap option
            print("[Step 3/5] Creating bootstrap option...")
            self.create_bootstrap_option()

            # Check funding alignment (warning only, doesn't stop execution)
            self._check_funding_alignment()

            # Step 4: Run deep hedge strategy
            print("[Step 4/5] Running deep hedge strategy...")
            deep_pnl = self.run_deep_hedge()

            # Step 5: Run BS baseline strategy
            print("[Step 5/5] Running BS baseline strategy...")
            bs_pnl = self.run_bs_baseline()

            # Create results object
            print("\nCreating results object...")
            results = BacktestResults(
                deep_pnl=deep_pnl,
                bs_pnl=bs_pnl,
                deep_positions=self.deep_positions,
                bs_positions=self.bs_positions,
                spots=self.option.underlier.spot,
                config=self.config,
            )

        except FileNotFoundError as e:
            print("\n" + "=" * 60)
            print("❌ BACKTEST FAILED: File Not Found")
            print("=" * 60)
            print(f"\nError: {e}")
            print("\nCommon causes:")
            print("  - Model checkpoint path is incorrect")
            print("  - Data directory doesn't exist or is empty")
            print("  - Missing perpetual data files (*perpetual*.parquet)")
            print("\nPlease check your configuration and file paths.")
            print("=" * 60 + "\n")
            raise

        except ValueError as e:
            print("\n" + "=" * 60)
            print("❌ BACKTEST FAILED: Invalid Data or Configuration")
            print("=" * 60)
            print(f"\nError: {e}")
            print("\nCommon causes:")
            print("  - Data directory is empty or has no matching files")
            print("  - Date range doesn't overlap with available data")
            print("  - Invalid configuration parameters")
            print("  - Bootstrap path generation failed")
            print("\nPlease check your data and configuration.")
            print("=" * 60 + "\n")
            raise

        except (RuntimeError, KeyError) as e:
            print("\n" + "=" * 60)
            print("❌ BACKTEST FAILED: Model or Execution Error")
            print("=" * 60)
            print(f"\nError: {e}")
            print("\nCommon causes:")
            print("  - Model checkpoint is corrupted or incompatible")
            print("  - Model architecture doesn't match checkpoint")
            print("  - CUDA/device mismatch")
            print("  - Tensor shape mismatch during computation")
            print("\nPlease check your model checkpoint and device settings.")
            print("=" * 60 + "\n")
            raise

        except Exception as e:
            print("\n" + "=" * 60)
            print("❌ BACKTEST FAILED: Unexpected Error")
            print("=" * 60)
            print(f"\nError type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\nPlease check the full traceback above for details.")
            print("=" * 60 + "\n")
            raise

        # Print final summary
        print("\n" + "=" * 60)
        print("BACKTEST COMPLETE")
        print("=" * 60)

        summary = results.summary()

        print("\nPERFORMANCE SUMMARY")
        print("-" * 60)

        # Deep Hedge metrics
        deep = summary["deep_hedge"]
        print(f"\nDeep Hedge:")
        print(f"  Mean PnL: ${deep['mean']:,.2f}")
        print(f"  Std PnL: ${deep['std']:,.2f}")
        print(f"  Sharpe Ratio: {deep['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {deep['sortino_ratio']:.3f}")
        print(f"  Max Drawdown: ${deep['max_drawdown']:.2f}")
        print(f"  CVaR (95%): ${deep['cvar_95']:.2f}")
        print(f"  Win Rate: {deep['win_rate']:.1%}")

        # BS Baseline metrics
        bs = summary["bs_baseline"]
        print(f"\nBlack-Scholes Baseline:")
        print(f"  Mean PnL: ${bs['mean']:,.2f}")
        print(f"  Std PnL: ${bs['std']:,.2f}")
        print(f"  Sharpe Ratio: {bs['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {bs['sortino_ratio']:.3f}")
        print(f"  Max Drawdown: ${bs['max_drawdown']:.2f}")
        print(f"  CVaR (95%): ${bs['cvar_95']:.2f}")
        print(f"  Win Rate: {bs['win_rate']:.1%}")

        # Comparison
        print(f"\nComparison (Deep Hedge vs BS):")
        mean_improvement = deep["mean"] - bs["mean"]
        sharpe_improvement = deep["sharpe_ratio"] - bs["sharpe_ratio"]
        print(f"  Mean PnL improvement: ${mean_improvement:+,.2f}")
        print(f"  Sharpe improvement: {sharpe_improvement:+.3f}")

        print("=" * 60 + "\n")

        return results

    def __repr__(self) -> str:
        """String representation."""
        return f"Backtester(config={self.config})"
