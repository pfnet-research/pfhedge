"""Backtesting framework for deep hedging strategies."""

from typing import Optional
import os
import pickle
import torch
from torch import Tensor

from .config import BacktestConfig


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

    def load_model(self, device: Optional[str] = None):
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

    def load_data(self):
        """Load historical data for backtesting.

        Loads perpetual and options data from parquet files using CryptoDataLoader,
        resamples to the configured frequency (dt_hours), and filters by date range.

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

        # Convert relative path to absolute if needed
        if not os.path.isabs(data_dir):
            # Assume relative to crypto/data directory
            base_dir = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(base_dir, "crypto", "data", data_dir)

        # Check if data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"(original path: {self.config.data_dir})"
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

        # Parse date range (keep as timezone-naive for perpetual data)
        try:
            start_date_naive = pd.to_datetime(self.config.start_date)
            end_date_naive = pd.to_datetime(self.config.end_date)
        except Exception as e:
            raise ValueError(f"Failed to parse dates: {e}")

        # Filter by date range
        print(
            f"Filtering data from {self.config.start_date} to {self.config.end_date}..."
        )
        mask = (resampled_df["timestamp"] >= start_date_naive) & (
            resampled_df["timestamp"] <= end_date_naive
        )
        filtered_df = resampled_df[mask].reset_index(drop=True)

        if filtered_df.empty:
            raise ValueError(
                f"No data found in date range [{self.config.start_date}, {self.config.end_date}]. "
                f"Available data range: [{resampled_df['timestamp'].min()}, {resampled_df['timestamp'].max()}]"
            )

        print(f"✅ Filtered to {len(filtered_df)} records in date range")

        # Update loader's perpetual_data with filtered data
        loader.perpetual_data = filtered_df

        # Load options data (optional, for real option price comparison)
        try:
            options_df = loader.load_options_data()

            # Filter options by date range too
            if not options_df.empty and "timestamp" in options_df.columns:
                # Options timestamps are timezone-aware (UTC), so make dates tz-aware for comparison
                if options_df["timestamp"].dt.tz is not None:
                    start_date_aware = start_date_naive.tz_localize("UTC")
                    end_date_aware = end_date_naive.tz_localize("UTC")
                    opts_mask = (options_df["timestamp"] >= start_date_aware) & (
                        options_df["timestamp"] <= end_date_aware
                    )
                else:
                    # Options timestamps are naive, use naive dates
                    opts_mask = (options_df["timestamp"] >= start_date_naive) & (
                        options_df["timestamp"] <= end_date_naive
                    )
                options_df = options_df[opts_mask].reset_index(drop=True)
                loader.options_data = options_df

            print(f"✅ Loaded {len(options_df)} options records in date range")
        except FileNotFoundError:
            print(f"⚠️  No options data found (this is okay for basic backtesting)")
            options_df = None

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

    def create_bootstrap_option(self, data_loader):
        """Create option with bootstrap paths from historical data.

        Args:
            data_loader: CryptoDataLoader with historical data

        Returns:
            BitcoinEuropeanOption with bootstrap paths

        Raises:
            NotImplementedError: To be implemented in Step 1.6
        """
        raise NotImplementedError(
            "create_bootstrap_option() will be implemented in Step 1.6"
        )

    def run_deep_hedge(self, option, model) -> Tensor:
        """Run deep hedging strategy on option.

        Args:
            option: Option to hedge
            model: Pre-trained Hedger model

        Returns:
            Cumulative PnL tensor, shape (n_paths, n_steps)

        Raises:
            NotImplementedError: To be implemented in Step 1.7
        """
        raise NotImplementedError("run_deep_hedge() will be implemented in Step 1.7")

    def run_bs_baseline(self, option) -> Tensor:
        """Run Black-Scholes delta hedge baseline.

        Args:
            option: Option to hedge

        Returns:
            Cumulative PnL tensor, shape (n_paths, n_steps)

        Raises:
            NotImplementedError: To be implemented in Step 1.8
        """
        raise NotImplementedError("run_bs_baseline() will be implemented in Step 1.8")

    def run(self):
        """Run full backtest.

        This orchestrates the entire backtesting process:
        1. Load model
        2. Load data
        3. Create bootstrap option
        4. Run deep hedge strategy
        5. Run BS baseline strategy
        6. Calculate metrics
        7. Create results object

        Returns:
            BacktestResults object with all results

        Raises:
            NotImplementedError: To be implemented in Step 1.10
        """
        raise NotImplementedError("run() will be implemented in Step 1.10")

    def __repr__(self) -> str:
        """String representation."""
        return f"Backtester(config={self.config})"
