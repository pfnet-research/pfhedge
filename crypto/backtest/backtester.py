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

        Returns:
            CryptoDataLoader with historical data

        Raises:
            NotImplementedError: To be implemented in Step 1.5
        """
        raise NotImplementedError("load_data() will be implemented in Step 1.5")

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
