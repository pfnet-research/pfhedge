"""Logging infrastructure for crypto hedging module.

This module provides centralized logging configuration for backtest and training pipelines.
Logs key configuration, resolved paths, and runtime information at INFO level.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logger(
    name: str = "crypto",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Set up a logger with consistent formatting.

    Args:
        name: Logger name (default: "crypto")
        level: Logging level (default: logging.INFO)
        log_file: Optional path to log file. If provided, logs are written to file.
        console: Whether to log to console (default: True)

    Returns:
        Configured logger instance

    Examples:
        >>> logger = setup_logger("crypto.backtest")
        >>> logger.info("Starting backtest")

        >>> logger = setup_logger("crypto.training", log_file="training.log")
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to root logger

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter with timestamp, name, level, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_file provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_config(
    logger: logging.Logger, config: Any, config_type: str = "Config"
) -> None:
    """Log configuration details at INFO level.

    Args:
        logger: Logger instance
        config: Configuration object (BacktestConfig or TrainingConfig)
        config_type: Type of config for display (default: "Config")

    Examples:
        >>> from crypto.backtest.config import BacktestConfig
        >>> config = BacktestConfig(...)
        >>> logger = setup_logger("crypto.backtest")
        >>> log_config(logger, config, "BacktestConfig")
    """
    logger.info("=" * 70)
    logger.info(f"{config_type} Configuration")
    logger.info("=" * 70)

    config_dict = config.to_dict()
    for key, value in config_dict.items():
        # Format different types appropriately
        if isinstance(value, float):
            if "cost" in key or "vol" in key:
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        elif isinstance(value, (int, bool, str)):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info("=" * 70)


def log_provenance(logger: logging.Logger, provenance: Dict[str, Any]) -> None:
    """Log provenance information for reproducibility.

    Args:
        logger: Logger instance
        provenance: Provenance dictionary from config.get_provenance_info()

    Examples:
        >>> from crypto.backtest.config import BacktestConfig
        >>> config = BacktestConfig.load_yaml("config.yaml")
        >>> provenance = config.get_provenance_info()
        >>> logger = setup_logger("crypto.backtest")
        >>> log_provenance(logger, provenance)
    """
    logger.info("Provenance Information:")
    logger.info("  Timestamp: %s", provenance.get("timestamp"))
    logger.info("  Python version: %s", provenance.get("python_version"))
    logger.info("  Platform: %s", provenance.get("platform"))

    if provenance.get("git_commit"):
        logger.info("  Git commit: %s", provenance["git_commit"][:8])
        if provenance.get("git_branch"):
            logger.info("  Git branch: %s", provenance["git_branch"])
        if provenance.get("git_dirty"):
            logger.warning("  Git status: DIRTY (uncommitted changes)")
        else:
            logger.info("  Git status: clean")

    resolved_paths = provenance.get("resolved_paths", {})
    if resolved_paths:
        logger.info("Resolved Paths:")
        for path_name, path_value in resolved_paths.items():
            if path_value:
                logger.info("  %s: %s", path_name, path_value)


def log_section_header(logger: logging.Logger, title: str, width: int = 70) -> None:
    """Log a section header for better readability.

    Args:
        logger: Logger instance
        title: Section title
        width: Width of separator line (default: 70)

    Examples:
        >>> logger = setup_logger("crypto")
        >>> log_section_header(logger, "Loading Data")
    """
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def log_step(
    logger: logging.Logger, step_num: int, total_steps: int, description: str
) -> None:
    """Log a numbered step in a multi-step process.

    Args:
        logger: Logger instance
        step_num: Current step number (1-indexed)
        total_steps: Total number of steps
        description: Step description

    Examples:
        >>> logger = setup_logger("crypto.backtest")
        >>> log_step(logger, 1, 5, "Loading model")
    """
    logger.info("[Step %d/%d] %s", step_num, total_steps, description)


def log_metrics(
    logger: logging.Logger, metrics: Dict[str, Any], prefix: str = ""
) -> None:
    """Log metrics dictionary at INFO level.

    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        prefix: Optional prefix for metric names

    Examples:
        >>> logger = setup_logger("crypto.backtest")
        >>> metrics = {"mean": 100.5, "std": 20.3, "sharpe_ratio": 1.5}
        >>> log_metrics(logger, metrics, "Deep Hedge")
    """
    prefix_str = f"{prefix} - " if prefix else ""
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{prefix_str}{key}: {value:.4f}")
        else:
            logger.info(f"{prefix_str}{key}: {value}")


def get_log_file_path(config: Any, pipeline_type: str) -> str:
    """Generate standard log file path based on config.

    Args:
        config: Configuration object (BacktestConfig or TrainingConfig)
        pipeline_type: Type of pipeline ("backtest" or "training")

    Returns:
        Path to log file in output directory

    Examples:
        >>> from crypto.backtest.config import BacktestConfig
        >>> config = BacktestConfig(output_dir="results")
        >>> path = get_log_file_path(config, "backtest")
        >>> print(path)  # results/backtest_20240101_120000.log
    """
    output_dir = Path(config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{pipeline_type}_{timestamp}.log"
    return str(output_dir / log_filename)
