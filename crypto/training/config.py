"""Configuration for deep hedging training."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for training a deep hedging model.

    Args:
        strike: Option strike price (absolute value, e.g., 50000)
        maturity_days: Option maturity in days
        call: True for call option, False for put option (default: True)
        volatility: Volatility for simulation (e.g., 0.8 for 80%)
        drift: Drift for simulation (default: 0.0)
        transaction_cost: Transaction cost rate, e.g., 0.0005 for 0.05% (default: 0.0005)
        dt_hours: Time step in hours (default: 8.0 for 8-hour rebalancing)
        n_paths: Number of simulation paths for training (default: 10000)
        n_epochs: Number of training epochs (default: 80)
        n_layers: Number of hidden layers in neural network (default: 4)
        n_units: Number of units per hidden layer (default: 128)
        risk_measure: Risk measure for training criterion (default: "expected_shortfall")
        risk_param: Parameter for risk measure, e.g., CVaR alpha (default: 0.9)
        model_path: Path to save trained model checkpoint (default: "models/deep_hedger_trained.pth")
        test_n_paths: Number of paths for test evaluation (default: 200)
        test_seed: Random seed for test set (default: 888)
        train_seed: Random seed for training (default: 42)
        output_dir: Directory to save training outputs (default: "training_results")

    Examples:
        >>> config = TrainingConfig(
        ...     strike=50000,
        ...     maturity_days=14,
        ...     volatility=0.8,
        ...     n_epochs=100
        ... )
        >>> config.validate()
        >>> config_dict = config.to_dict()
    """

    # Option parameters
    strike: float
    maturity_days: int

    # Optional parameters with defaults
    call: bool = True

    # Market simulation parameters
    volatility: float = 0.8
    drift: float = 0.0
    transaction_cost: float = 0.0005
    dt_hours: float = 8.0

    # Training parameters
    n_paths: int = 10000
    n_epochs: int = 80

    # Model architecture
    n_layers: int = 4
    n_units: int = 128
    risk_measure: str = "expected_shortfall"
    risk_param: float = 0.9

    # Device
    device: str = "cpu"

    # Model output
    model_path: str = "models/deep_hedger_trained.pth"

    # Testing parameters
    test_n_paths: int = 200
    test_seed: int = 888
    train_seed: int = 42

    # Output directory
    output_dir: str = "training_results"

    def normalize_risk_measure(self) -> str:
        """Normalize risk measure to canonical form.

        Maps aliases to canonical names:
        - "cvar" -> "expected_shortfall"
        - All others pass through

        Returns:
            Canonical risk measure name
        """
        # Map aliases to canonical names
        alias_map = {
            "cvar": "expected_shortfall",
            "es": "expected_shortfall",
        }
        return alias_map.get(self.risk_measure.lower(), self.risk_measure)

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate strike
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")

        # Validate maturity
        if self.maturity_days <= 0:
            raise ValueError(
                f"maturity_days must be positive, got {self.maturity_days}"
            )

        # Validate volatility
        if self.volatility <= 0:
            raise ValueError(f"volatility must be positive, got {self.volatility}")
        if self.volatility > 5.0:
            raise ValueError(
                f"volatility seems too high: {self.volatility}. Did you mean {self.volatility/100}?"
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

        # Validate training parameters
        if self.n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {self.n_paths}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")

        # Validate model architecture
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_units <= 0:
            raise ValueError(f"n_units must be positive, got {self.n_units}")

        # Normalize and validate risk measure
        # First normalize aliases (cvar -> expected_shortfall, etc.)
        normalized_measure = self.normalize_risk_measure()
        valid_measures = ["expected_shortfall", "variance", "entropic"]
        if normalized_measure not in valid_measures:
            raise ValueError(
                f"risk_measure must be one of {valid_measures} (or aliases: cvar, es), got '{self.risk_measure}'"
            )
        # Update to normalized form for consistency
        self.risk_measure = normalized_measure

        # Validate risk_param
        if self.risk_param <= 0 or self.risk_param > 1:
            raise ValueError(f"risk_param must be in (0, 1], got {self.risk_param}")

        # Validate test parameters
        if self.test_n_paths <= 0:
            raise ValueError(f"test_n_paths must be positive, got {self.test_n_paths}")

        # Validate device
        valid_devices = ["cpu", "cuda", "mps"]
        device_base = self.device.split(":")[0]  # Handle "cuda:0", "cuda:1", etc.
        if device_base not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices} (or cuda:N), got '{self.device}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary.

        Args:
            config_dict: Dictionary with config parameters

        Returns:
            TrainingConfig instance

        Examples:
            >>> config_dict = {
            ...     "strike": 50000,
            ...     "maturity_days": 14,
            ...     "volatility": 0.8
            ... }
            >>> config = TrainingConfig.from_dict(config_dict)
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
            f"TrainingConfig(\n"
            f"  option: {'Call' if self.call else 'Put'} @ ${self.strike:,.0f}, {self.maturity_days}d\n"
            f"  market: vol={self.volatility:.1%}, drift={self.drift:.3f}, cost={self.transaction_cost:.2%}\n"
            f"  training: {self.n_paths:,} paths, {self.n_epochs} epochs, seed={self.train_seed}\n"
            f"  model: {self.n_layers}×{self.n_units} units, {self.risk_measure}(p={self.risk_param})\n"
            f"  output: {self.model_path}\n"
            f")"
        )
