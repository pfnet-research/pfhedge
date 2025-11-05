"""Configuration for deep hedging training."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union, List


@dataclass
class TrainingConfig:
    """Configuration for training a deep hedging model.

    Args:
        strike: Option strike price (absolute value, e.g., 50000)
        maturity_days: Option maturity in days
        call: True for call option, False for put option (default: True)
        volatility: Volatility for simulation (e.g., 0.8 for 80%)
        volatility_window: Rolling window size for realized volatility (default: 20, 0 = use constant vol)
        drift: Drift for simulation (default: 0.0)
        transaction_cost: Transaction cost rate, e.g., 0.0005 for 0.05% (default: 0.0005)
        dt_hours: Time step in hours (default: 8.0 for 8-hour rebalancing)
        n_paths: Number of simulation paths for training (default: 10000)
        n_epochs: Number of training epochs (default: 80)
        n_layers: Number of hidden layers in neural network (default: 4)
        n_units: Number of units per hidden layer (default: 128)
        risk_measure: Risk measure for training criterion (default: "expected_shortfall")
        risk_param: Parameter for risk measure, e.g., CVaR alpha (default: 0.9)
        model_path: Path to save trained model checkpoint (required, no default)
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

    # Option parameters (required)
    strike: float
    maturity_days: int
    model_path: str  # Path to save trained model checkpoint (required)

    # Optional parameters with defaults
    call: bool = True

    # Market simulation parameters
    volatility: float = 0.8
    volatility_window: int = (
        20  # Rolling window for realized volatility (0 = use constant vol)
    )
    drift: float = 0.0
    transaction_cost: float = 0.0005
    dt_hours: float = 8.0
    underlying_type: str = "perpetual"  # "perpetual" or "spot"

    # Training parameters
    n_paths: int = 10000
    n_epochs: int = 80

    # Model architecture
    model_type: str = "mlp"  # "mlp", "lstm", or "gru"
    n_layers: int = 4
    n_units: Union[int, List[int]] = 128  # Single int or list for variable layer sizes
    risk_measure: str = "expected_shortfall"
    risk_param: float = 0.9

    # Device
    device: str = "cpu"

    # Optimizer settings
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Early stopping
    early_stopping: bool = False
    patience: int = 10  # Number of epochs to wait for improvement
    min_delta: float = 1e-6  # Minimum change to qualify as improvement

    # Memory optimizations
    use_amp: bool = True  # Use mixed precision training (FP16/BF16) on CUDA
    validation_freq: int = 1  # Validation frequency (1=every epoch, >1=every N epochs)

    # Testing parameters
    test_n_paths: int = 200
    test_seed: int = 888
    train_seed: int = 42

    # Output directory
    output_dir: str = "training_results"

    # Model features
    features: Optional[List[str]] = None  # None = use DEFAULT_FEATURES

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

        # Validate underlying_type
        valid_underlying_types = ["perpetual", "spot"]
        if self.underlying_type not in valid_underlying_types:
            raise ValueError(
                f"underlying_type must be one of {valid_underlying_types}, got '{self.underlying_type}'"
            )

        # Validate training parameters
        if self.n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {self.n_paths}")
        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.n_epochs}")

        # Validate model architecture
        valid_model_types = ["mlp", "lstm", "gru"]
        if self.model_type.lower() not in valid_model_types:
            raise ValueError(
                f"model_type must be one of {valid_model_types}, got '{self.model_type}'"
            )
        self.model_type = self.model_type.lower()  # Normalize to lowercase

        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")

        # Validate n_units (can be int or list of ints)
        if isinstance(self.n_units, int):
            if self.n_units <= 0:
                raise ValueError(f"n_units must be positive, got {self.n_units}")
        elif isinstance(self.n_units, list):
            if len(self.n_units) != self.n_layers:
                raise ValueError(
                    f"When n_units is a list, it must have {self.n_layers} elements (one per layer), "
                    f"got {len(self.n_units)}: {self.n_units}"
                )
            for i, units in enumerate(self.n_units):
                if units <= 0:
                    raise ValueError(f"n_units[{i}] must be positive, got {units}")
        else:
            raise ValueError(
                f"n_units must be int or list of ints, got {type(self.n_units).__name__}"
            )

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

        # Validate risk_param based on risk measure
        if self.risk_param <= 0:
            raise ValueError(f"risk_param must be positive, got {self.risk_param}")

        # Additional constraints for specific risk measures
        if normalized_measure == "expected_shortfall" and self.risk_param > 1:
            raise ValueError(
                f"risk_param for expected_shortfall must be in (0, 1], got {self.risk_param}"
            )

        # Validate optimizer
        valid_optimizers = ["adam", "adamw", "sgd"]
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(
                f"optimizer must be one of {valid_optimizers}, got '{self.optimizer}'"
            )
        self.optimizer = self.optimizer.lower()

        # Validate optimizer hyperparameters
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )

        # Validate early stopping parameters
        if self.early_stopping:
            if self.patience <= 0:
                raise ValueError(
                    f"patience must be positive when early_stopping=True, got {self.patience}"
                )
            if self.min_delta < 0:
                raise ValueError(
                    f"min_delta must be non-negative, got {self.min_delta}"
                )

        # Validate test parameters
        if self.test_n_paths <= 0:
            raise ValueError(f"test_n_paths must be positive, got {self.test_n_paths}")

        # Validate device (syntax only - availability checked when actually used)
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

    def get_provenance_info(self) -> Dict[str, Any]:
        """Get provenance information for reproducibility.

        Returns:
            Dictionary with provenance information including:
            - config: Full config as dict
            - git_commit: Git commit hash if available
            - git_branch: Git branch if available
            - git_dirty: Whether repo has uncommitted changes
            - python_version: Python version string
            - platform: Operating system
            - timestamp: Current timestamp

        Examples:
            >>> config = TrainingConfig(strike=50000, maturity_days=14, model_path="model.pth")
            >>> provenance = config.get_provenance_info()
            >>> print(provenance['git_commit'])
        """
        import subprocess
        import sys
        import platform
        from datetime import datetime

        provenance = {
            "config": self.to_dict(),
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
            "timestamp": datetime.now().isoformat(),
        }

        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            provenance["git_commit"] = git_commit
        except (subprocess.CalledProcessError, FileNotFoundError):
            provenance["git_commit"] = None

        try:
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            provenance["git_branch"] = git_branch
        except (subprocess.CalledProcessError, FileNotFoundError):
            provenance["git_branch"] = None

        try:
            git_dirty = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            provenance["git_dirty"] = bool(git_dirty)
        except (subprocess.CalledProcessError, FileNotFoundError):
            provenance["git_dirty"] = None

        return provenance

    def compute_config_hash(self) -> str:
        """Compute unique hash for this training configuration.

        This hash uniquely identifies the training configuration,
        useful for tracking experiments and ensuring reproducibility.

        Returns:
            16-character hex hash string

        Examples:
            >>> config = TrainingConfig(strike=50000, maturity_days=14, model_path="model.pth")
            >>> hash1 = config.compute_config_hash()
            >>> print(hash1)
        """
        import hashlib
        import json

        provenance = self.get_provenance_info()

        components = []

        # Config (sorted for consistency)
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        components.append(config_str)

        # Git commit (if available)
        if provenance["git_commit"]:
            components.append(f"git:{provenance['git_commit'][:8]}")

        # Platform and Python version
        components.append(f"platform:{provenance['platform']}")
        components.append(f"python:{provenance['python_version']}")

        combined = "|".join(components)
        full_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        return full_hash[:16]

    def __repr__(self) -> str:
        """String representation."""
        # Format n_units for display
        if isinstance(self.n_units, list):
            units_str = f"[{', '.join(str(u) for u in self.n_units)}]"
        else:
            units_str = f"{self.n_layers}×{self.n_units}"

        return (
            f"TrainingConfig(\n"
            f"  option: {'Call' if self.call else 'Put'} @ ${self.strike:,.0f}, {self.maturity_days}d\n"
            f"  market: vol={self.volatility:.1%}, drift={self.drift:.3f}, cost={self.transaction_cost:.2%}\n"
            f"  training: {self.n_paths:,} paths, {self.n_epochs} epochs, seed={self.train_seed}\n"
            f"  model: {units_str} units, {self.risk_measure}(p={self.risk_param})\n"
            f"  output: {self.model_path}\n"
            f")"
        )
