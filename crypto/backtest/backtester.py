"""Backtesting framework for deep hedging strategies."""

from typing import Optional
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

    def load_model(self):
        """Load pre-trained model from checkpoint.

        Returns:
            Loaded Hedger model

        Raises:
            NotImplementedError: To be implemented in Step 1.4
        """
        raise NotImplementedError("load_model() will be implemented in Step 1.4")

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
