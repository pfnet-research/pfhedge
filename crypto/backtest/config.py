"""Configuration for backtesting."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


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

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file

        Raises:
            ImportError: If PyYAML is not installed

        Examples:
            >>> config = BacktestConfig(
            ...     start_date="2024-01-01",
            ...     end_date="2024-01-31",
            ...     strike=50000,
            ...     maturity_days=14,
            ...     model_path="models/model.pth"
            ... )
            >>> config.save_yaml("config.yaml")
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install with: pip install pyyaml"
            )

        config_dict = self.to_dict()
        path_obj = Path(path)

        # Create parent directory if it doesn't exist
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w") as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    @classmethod
    def load_yaml(
        cls, path: str, anchor_relative_paths: bool = True
    ) -> "BacktestConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file
            anchor_relative_paths: If True, resolve relative paths relative to config file directory (default: True)

        Returns:
            BacktestConfig instance

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If file does not exist
            ValueError: If YAML is invalid, missing required fields, or has unknown keys

        Examples:
            >>> config = BacktestConfig.load_yaml("config.yaml")
            >>> config.validate()
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install with: pip install pyyaml"
            )

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path_obj, "r") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            raise ValueError(f"Empty or invalid YAML file: {path}")

        # Check for unknown keys
        import inspect

        valid_fields = set(inspect.signature(cls).parameters.keys())
        provided_fields = set(config_dict.keys())
        unknown_fields = provided_fields - valid_fields

        if unknown_fields:
            raise ValueError(
                f"Unknown configuration fields: {', '.join(sorted(unknown_fields))}. "
                f"Valid fields are: {', '.join(sorted(valid_fields))}"
            )

        # Expand environment variables and tilde in paths
        import os

        for field in ["model_path", "data_dir", "output_dir"]:
            if field in config_dict and isinstance(config_dict[field], str):
                # First expand env vars, then tilde
                expanded = os.path.expandvars(config_dict[field])
                config_dict[field] = str(Path(expanded).expanduser())

        # Resolve relative paths relative to config file directory
        if anchor_relative_paths:
            config_dir = path_obj.parent.absolute()
            for field in ["model_path", "data_dir", "output_dir"]:
                if field in config_dict and isinstance(config_dict[field], str):
                    field_path = Path(config_dict[field])
                    # Only resolve if relative path
                    if not field_path.is_absolute():
                        resolved = (config_dir / field_path).resolve()
                        config_dict[field] = str(resolved)

        try:
            config = cls.from_dict(config_dict)
            # Validate config after loading
            config.validate()
            return config
        except TypeError as e:
            raise ValueError(
                f"Invalid YAML structure: {e}. "
                f"Make sure all required fields are present."
            )

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
            - resolved_paths: Absolute paths after resolution
            - git_commit: Git commit hash if available
            - git_branch: Git branch if available
            - git_dirty: Whether repo has uncommitted changes
            - python_version: Python version string
            - platform: Operating system
            - timestamp: Current timestamp

        Examples:
            >>> config = BacktestConfig.load_yaml("config.yaml")
            >>> provenance = config.get_provenance_info()
            >>> print(provenance['git_commit'])  # Git hash for reproducibility
        """
        import subprocess
        import sys
        import platform
        from datetime import datetime

        provenance = {
            "config": self.to_dict(),
            "resolved_paths": {
                "model_path": str(Path(self.model_path).absolute())
                if self.model_path
                else None,
                "data_dir": str(Path(self.data_dir).absolute()),
                "output_dir": str(Path(self.output_dir).absolute()),
            },
            "python_version": sys.version.split()[0],  # e.g., "3.10.16"
            "platform": platform.system(),  # e.g., "Darwin", "Linux", "Windows"
            "timestamp": datetime.now().isoformat(),
        }

        # Try to get git info
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
