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

    # Date range
    start_date: str
    end_date: str

    # Option parameters
    strike: float
    maturity_days: int
    model_path: str

    # Optional parameters with defaults
    call: bool = True
    n_bootstrap_paths: int = 100  # 0 = auto-use max unique paths from historical data
    transaction_cost: float = 0.0005
    dt_hours: float = 8.0
    volatility_window: int = (
        20  # Rolling window for realized volatility (0 = use constant vol)
    )
    underlying_type: str = "perpetual"  # "perpetual" or "spot" (must match training)
    band_width: float = 0.001  # Minimum trade size in BTC (0.0 = no filtering)
    data_dir: str = "sample_data"
    data_file: Optional[str] = (
        None  # Specific data file to load (overrides auto-detection)
    )
    output_dir: str = "backtest_results"

    # Bootstrap configuration (NEW)
    bootstrap_mode: str = "absolute_strike"
    initial_spot: Optional[float] = None
    target_moneyness: Optional[float] = None
    spot_tolerance: Optional[float] = 0.1

    # Diagnostics configuration
    enable_diagnostics: bool = False

    # Execution parameters
    seed: Optional[int] = None
    save_raw_data: bool = False

    def validate(self) -> None:
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

        # Validate n_bootstrap_paths (0 = auto, uses max unique paths)
        if self.n_bootstrap_paths < 0:
            raise ValueError(
                f"n_bootstrap_paths must be non-negative (0 = auto), got {self.n_bootstrap_paths}"
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

        # Validate bootstrap_mode
        valid_modes = ["absolute_strike", "normalize_spot"]
        if self.bootstrap_mode not in valid_modes:
            raise ValueError(
                f"bootstrap_mode must be one of {valid_modes}, got '{self.bootstrap_mode}'"
            )

        # Validate underlying_type
        valid_underlying_types = ["perpetual", "spot"]
        if self.underlying_type not in valid_underlying_types:
            raise ValueError(
                f"underlying_type must be one of {valid_underlying_types}, got '{self.underlying_type}'"
            )

        # Validate initial_spot if provided
        if self.initial_spot is not None:
            if self.initial_spot <= 0:
                raise ValueError(
                    f"initial_spot must be positive, got {self.initial_spot}"
                )

        # Validate target_moneyness if provided
        if self.target_moneyness is not None:
            if self.target_moneyness <= 0:
                raise ValueError(
                    f"target_moneyness must be positive, got {self.target_moneyness}"
                )

        # If normalize_spot, require initial_spot or target_moneyness
        if self.bootstrap_mode == "normalize_spot":
            if self.initial_spot is None and self.target_moneyness is None:
                raise ValueError(
                    "bootstrap_mode='normalize_spot' requires either 'initial_spot' or 'target_moneyness'.\n"
                    "Add 'initial_spot' from Step 1 option discovery to your config."
                )

        # Warn if both provided and they conflict
        if self.initial_spot is not None and self.target_moneyness is not None:
            calculated_moneyness = self.initial_spot / self.strike
            if abs(calculated_moneyness - self.target_moneyness) > 0.001:
                import warnings

                warnings.warn(
                    f"initial_spot ({self.initial_spot}) implies moneyness={calculated_moneyness:.4f} "
                    f"but target_moneyness={self.target_moneyness:.4f}. Using target_moneyness."
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BacktestConfig":
        return cls(**config_dict)

    def save_yaml(self, path: str) -> None:
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
        return self.dt_hours / 24 / 365

    @property
    def effective_target_moneyness(self) -> Optional[float]:
        # Handle divide-by-zero
        if self.strike == 0:
            raise ValueError("Cannot calculate moneyness with strike=0")

        if self.target_moneyness is not None:
            return self.target_moneyness
        elif self.initial_spot is not None:
            return self.initial_spot / self.strike
        return None

    def get_provenance_info(self) -> Dict[str, Any]:
        import subprocess
        import sys
        import platform
        from datetime import datetime

        provenance = {
            "config": self.to_dict(),
            "resolved_paths": {
                "model_path": (
                    str(Path(self.model_path).absolute()) if self.model_path else None
                ),
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

    def compute_provenance_hash(self) -> str:
        import hashlib
        import json

        provenance = self.get_provenance_info()

        # Build components for hash
        components = []

        # 1. Config (sorted for consistency)
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        components.append(config_str)

        # 2. Model file hash (if exists)
        model_path = Path(self.model_path)
        if model_path.exists() and model_path.is_file():
            try:
                with open(model_path, "rb") as f:
                    # Read in chunks for large files
                    file_hasher = hashlib.sha256()
                    while chunk := f.read(8192):
                        file_hasher.update(chunk)
                    model_hash = file_hasher.hexdigest()[:16]
                    components.append(f"model:{model_hash}")
            except (OSError, IOError):
                # File not readable, skip
                pass

        # 3. Git commit (if available)
        if provenance["git_commit"]:
            components.append(f"git:{provenance['git_commit'][:8]}")

        # 4. Platform and Python version (for reproducibility tracking)
        components.append(f"platform:{provenance['platform']}")
        components.append(f"python:{provenance['python_version']}")

        # Combine all components and hash
        combined = "|".join(components)
        full_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        # Return first 16 characters (sufficient for uniqueness in practice)
        return full_hash[:16]

    def __repr__(self) -> str:
        return (
            f"BacktestConfig(\n"
            f"  date_range: {self.start_date} to {self.end_date}\n"
            f"  option: {'Call' if self.call else 'Put'} @ ${self.strike:,.0f}, {self.maturity_days}d\n"
            f"  model: {self.model_path}\n"
            f"  execution: {self.n_bootstrap_paths} paths, {self.transaction_cost:.2%} cost, {self.dt_hours}h steps\n"
            f")"
        )
