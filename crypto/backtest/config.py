"""Configuration for backtesting."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        strike: Option strike price (absolute value, e.g., 50000)
        maturity_days: Option maturity in days
        model_path: Path to pre-trained model checkpoint (.pth file)
        call: True for call option, False for put option (default: True)
        n_bootstrap_paths: Number of bootstrap paths to generate (default: 100)
        transaction_cost: Transaction cost rate, e.g., 0.0005 for 0.05% (default: 0.0005)
        dt_hours: Time step in hours (default: 8.0 for 8-hour rebalancing)
        data_dir: Directory containing historical data (default: "sample_data")
        output_dir: Directory to save results (default: "backtest_results")

    Examples:
        >>> config = BacktestConfig(
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     strike=50000,
        ...     maturity_days=14,
        ...     model_path="models/deep_hedger.pth"
        ... )
        >>> config.validate()
        >>> config_dict = config.to_dict()
    """

    # Date range
    start_date: str
    end_date: str

    # Option parameters
    strike: float
    maturity_days: int
    model_path: str

    # Optional parameters with defaults
    call: bool = True
    n_bootstrap_paths: int = 100
    transaction_cost: float = 0.0005
    dt_hours: float = 8.0
    data_dir: str = "sample_data"
    output_dir: str = "backtest_results"

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate dates
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            if end <= start:
                raise ValueError(
                    f"end_date ({self.end_date}) must be after start_date ({self.start_date})"
                )
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError(
                    f"Dates must be in YYYY-MM-DD format. Got: {self.start_date}, {self.end_date}"
                )
            raise

        # Validate strike
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")

        # Validate maturity
        if self.maturity_days <= 0:
            raise ValueError(
                f"maturity_days must be positive, got {self.maturity_days}"
            )

        # Validate n_bootstrap_paths
        if self.n_bootstrap_paths <= 0:
            raise ValueError(
                f"n_bootstrap_paths must be positive, got {self.n_bootstrap_paths}"
            )

        # Validate transaction_cost
        if self.transaction_cost < 0:
            raise ValueError(
                f"transaction_cost must be non-negative, got {self.transaction_cost}"
            )
        if self.transaction_cost > 0.1:
            raise ValueError(
                f"transaction_cost seems too high: {self.transaction_cost} (10%+). Did you mean {self.transaction_cost/100}?"
            )

        # Validate dt_hours
        if self.dt_hours <= 0:
            raise ValueError(f"dt_hours must be positive, got {self.dt_hours}")
        if self.dt_hours > 24:
            raise ValueError(f"dt_hours must be <= 24, got {self.dt_hours}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BacktestConfig":
        """Create config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            BacktestConfig instance

        Examples:
            >>> config_dict = {
            ...     "start_date": "2024-01-01",
            ...     "end_date": "2024-01-31",
            ...     "strike": 50000,
            ...     "maturity_days": 14,
            ...     "model_path": "models/model.pth"
            ... }
            >>> config = BacktestConfig.from_dict(config_dict)
        """
        return cls(**config_dict)

    @property
    def dt(self) -> float:
        """Get dt in years (for compatibility with PFHedge).

        Returns:
            Time step in years
        """
        return self.dt_hours / 24 / 365

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BacktestConfig(\n"
            f"  date_range: {self.start_date} to {self.end_date}\n"
            f"  option: {'Call' if self.call else 'Put'} @ ${self.strike:,.0f}, {self.maturity_days}d\n"
            f"  model: {self.model_path}\n"
            f"  execution: {self.n_bootstrap_paths} paths, {self.transaction_cost:.2%} cost, {self.dt_hours}h steps\n"
            f")"
        )
