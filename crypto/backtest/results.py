"""Results container for backtesting framework."""

from typing import Dict, Optional, Any
import json
from datetime import datetime
from pathlib import Path
import torch
from torch import Tensor
import numpy as np

from .metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_cvar,
    calculate_var,
    calculate_win_rate,
    calculate_calmar_ratio,
)


class BacktestResults:
    """Container for backtest results with summary statistics.

    Stores PnL data, positions, and spot prices from deep hedge and
    Black-Scholes baseline strategies. Provides methods to calculate
    summary statistics and export data.

    **Important**: All tensor data is assumed to be immutable after construction.
    Summary statistics are cached for performance. If you modify any tensor fields
    after construction, you must manually reset `_summary_cache = None` to force
    recomputation.

    Args:
        deep_pnl: Deep hedging cumulative PnL, shape (n_paths, n_steps)
        bs_pnl: Black-Scholes cumulative PnL, shape (n_paths, n_steps)
        deep_positions: Deep hedge positions, shape (n_paths, n_steps)
        bs_positions: BS delta positions, shape (n_paths, n_steps)
        spots: Underlying spot prices, shape (n_paths, n_steps)
        config: Optional backtest configuration

    Examples:
        >>> results = BacktestResults(
        ...     deep_pnl=deep_pnl_tensor,
        ...     bs_pnl=bs_pnl_tensor,
        ...     deep_positions=deep_positions,
        ...     bs_positions=bs_positions,
        ...     spots=spot_prices
        ... )
        >>> summary = results.summary()
        >>> print(summary['deep_hedge']['sharpe_ratio'])
    """

    def __init__(
        self,
        deep_pnl: Tensor,
        bs_pnl: Tensor,
        deep_positions: Tensor,
        bs_positions: Tensor,
        spots: Tensor,
        config: Optional[object] = None,
    ):
        """Initialize results container.

        Args:
            deep_pnl: Deep hedging cumulative PnL (n_paths, n_steps)
            bs_pnl: Black-Scholes cumulative PnL (n_paths, n_steps)
            deep_positions: Deep hedge positions (n_paths, n_steps)
            bs_positions: BS delta positions (n_paths, n_steps)
            spots: Spot prices (n_paths, n_steps)
            config: Optional BacktestConfig object
        """
        # Validate inputs
        self._validate_inputs(deep_pnl, bs_pnl, deep_positions, bs_positions, spots)

        # Store data
        self.deep_pnl = deep_pnl
        self.bs_pnl = bs_pnl
        self.deep_positions = deep_positions
        self.bs_positions = bs_positions
        self.spots = spots
        self.config = config

        # Derived properties
        self.n_paths = deep_pnl.shape[0]
        self.n_steps = deep_pnl.shape[1]

        # Cache for summary statistics (computed once on first access)
        self._summary_cache: Optional[Dict[str, Dict[str, float]]] = None

    def _validate_inputs(
        self,
        deep_pnl: Tensor,
        bs_pnl: Tensor,
        deep_positions: Tensor,
        bs_positions: Tensor,
        spots: Tensor,
    ) -> None:
        """Validate that all inputs have consistent shapes."""
        # Check all are 2D tensors
        for name, tensor in [
            ("deep_pnl", deep_pnl),
            ("bs_pnl", bs_pnl),
            ("deep_positions", deep_positions),
            ("bs_positions", bs_positions),
            ("spots", spots),
        ]:
            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} must be 2D tensor (n_paths, n_steps), got {tensor.ndim}D"
                )

        # Check all have same shape
        expected_shape = deep_pnl.shape
        for name, tensor in [
            ("bs_pnl", bs_pnl),
            ("deep_positions", deep_positions),
            ("bs_positions", bs_positions),
            ("spots", spots),
        ]:
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {tensor.shape} doesn't match deep_pnl shape {expected_shape}"
                )

    def summary(
        self, alpha_cvar: float = 0.05, alpha_var: float = 0.05
    ) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for both strategies.

        Computes comprehensive performance metrics for deep hedge and
        Black-Scholes baseline, including:
        - Basic stats: mean, std, min, max, median
        - Risk-adjusted: Sharpe ratio, Sortino ratio
        - Risk metrics: CVaR, VaR, max drawdown, Calmar ratio
        - Win rate (percentage of paths with final PnL > 0)

        Note: Results are cached after first computation to avoid expensive
        recomputation on subsequent calls (e.g., in __repr__).

        Args:
            alpha_cvar: Significance level for CVaR (default 0.05 = 95% CVaR)
            alpha_var: Significance level for VaR (default 0.05 = 95% VaR)

        Returns:
            Dictionary with 'deep_hedge' and 'bs_baseline' keys, each containing
            metrics dictionary

        Examples:
            >>> summary = results.summary()
            >>> print(f"Deep Sharpe: {summary['deep_hedge']['sharpe_ratio']:.3f}")
            >>> print(f"BS Sharpe: {summary['bs_baseline']['sharpe_ratio']:.3f}")
        """
        # Return cached summary if available and alpha values match default
        if self._summary_cache is not None and alpha_cvar == 0.05 and alpha_var == 0.05:
            return self._summary_cache

        # Get final PnL for each strategy
        deep_final = self.deep_pnl[:, -1]
        bs_final = self.bs_pnl[:, -1]

        # Calculate metrics for deep hedge
        deep_metrics = self._calculate_strategy_metrics(
            final_pnl=deep_final,
            cum_pnl=self.deep_pnl,
            alpha_cvar=alpha_cvar,
            alpha_var=alpha_var,
        )

        # Calculate metrics for BS baseline
        bs_metrics = self._calculate_strategy_metrics(
            final_pnl=bs_final,
            cum_pnl=self.bs_pnl,
            alpha_cvar=alpha_cvar,
            alpha_var=alpha_var,
        )

        result = {"deep_hedge": deep_metrics, "bs_baseline": bs_metrics}

        # Cache if using default alpha values
        if alpha_cvar == 0.05 and alpha_var == 0.05:
            self._summary_cache = result

        return result

    def _calculate_strategy_metrics(
        self, final_pnl: Tensor, cum_pnl: Tensor, alpha_cvar: float, alpha_var: float
    ) -> Dict[str, float]:
        """Calculate metrics for a single strategy."""
        metrics = {}

        # Basic statistics
        metrics["mean"] = final_pnl.mean().item()
        metrics["std"] = final_pnl.std().item()
        metrics["min"] = final_pnl.min().item()
        metrics["max"] = final_pnl.max().item()
        metrics["median"] = final_pnl.median().item()

        # Risk-adjusted metrics
        metrics["sharpe_ratio"] = calculate_sharpe_ratio(final_pnl)
        metrics["sortino_ratio"] = calculate_sortino_ratio(final_pnl)

        # Risk metrics
        cvar_pct = int((1 - alpha_cvar) * 100)
        var_pct = int((1 - alpha_var) * 100)
        metrics[f"cvar_{cvar_pct}"] = calculate_cvar(final_pnl, alpha=alpha_cvar)
        metrics[f"var_{var_pct}"] = calculate_var(final_pnl, alpha=alpha_var)
        metrics["max_drawdown"] = calculate_max_drawdown(cum_pnl)
        metrics["calmar_ratio"] = calculate_calmar_ratio(cum_pnl)

        # Win rate
        metrics["win_rate"] = calculate_win_rate(final_pnl)

        return metrics

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        """Export all data to dictionary.

        Args:
            include_raw: If True (default), includes raw tensor data as lists.
                         If False, includes only summary statistics and metadata.
                         Set to False for large datasets to reduce memory usage.

        Returns:
            Dictionary containing:
            - 'n_paths': Number of simulation paths
            - 'n_steps': Number of time steps
            - 'summary': Summary statistics
            - 'config': Config dict if available
            - 'deep_pnl': Deep hedge PnL as list (if include_raw=True)
            - 'bs_pnl': BS baseline PnL as list (if include_raw=True)
            - 'deep_positions': Deep hedge positions as list (if include_raw=True)
            - 'bs_positions': BS positions as list (if include_raw=True)
            - 'spots': Spot prices as list (if include_raw=True)

        Examples:
            >>> # Full export with raw data
            >>> data = results.to_dict()
            >>>
            >>> # Lightweight export without raw arrays
            >>> summary_only = results.to_dict(include_raw=False)
            >>>
            >>> # Save to JSON
            >>> import json
            >>> with open('results.json', 'w') as f:
            ...     json.dump(data, f)
        """
        data = {
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "summary": self.summary(),
        }

        # Add raw tensor data if requested
        if include_raw:
            data["deep_pnl"] = self.deep_pnl.tolist()
            data["bs_pnl"] = self.bs_pnl.tolist()
            data["deep_positions"] = self.deep_positions.tolist()
            data["bs_positions"] = self.bs_positions.tolist()
            data["spots"] = self.spots.tolist()

        # Add config if available
        if self.config is not None:
            if hasattr(self.config, "to_dict"):
                data["config"] = self.config.to_dict()
            else:
                data["config"] = str(self.config)

        return data

    @staticmethod
    def _safe_json_normalize(obj: Any) -> Any:
        """Recursively normalize data for JSON serialization.

        Handles common non-JSON types:
        - datetime objects -> ISO format strings
        - numpy types -> Python native types
        - Path objects -> strings
        - Recursively processes dicts and lists

        Args:
            obj: Object to normalize

        Returns:
            JSON-serializable version of obj
        """
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (Path,)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: BacktestResults._safe_json_normalize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BacktestResults._safe_json_normalize(item) for item in obj]
        else:
            return obj

    def to_json(
        self, filepath: Optional[str] = None, include_raw: bool = False, **kwargs
    ) -> Optional[str]:
        """Export results to JSON format.

        Convenience method that handles non-JSON types (dates, numpy types)
        and optionally writes to file.

        Args:
            filepath: Optional path to write JSON file. If None, returns JSON string.
            include_raw: If True, includes raw tensor data. If False, only metadata
                         and summary (default False for smaller file size).
            **kwargs: Additional arguments passed to json.dumps() (e.g., indent=2)

        Returns:
            JSON string if filepath is None, otherwise None (writes to file)

        Examples:
            >>> # Get JSON string without raw data
            >>> json_str = results.to_json(include_raw=False)
            >>>
            >>> # Write to file with pretty formatting
            >>> results.to_json('results.json', include_raw=True, indent=2)
            >>>
            >>> # Lightweight export (summary only)
            >>> results.to_json('summary.json', include_raw=False, indent=2)
        """
        # Get dictionary and normalize for JSON
        data = self.to_dict(include_raw=include_raw)
        normalized = self._safe_json_normalize(data)

        # Set default indent if not specified
        if "indent" not in kwargs:
            kwargs["indent"] = 2

        # Write to file or return string
        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(normalized, f, **kwargs)
            return None
        else:
            return json.dumps(normalized, **kwargs)

    def __repr__(self) -> str:
        """String representation.

        Note: Uses cached summary if available, avoiding expensive recomputation.
        """
        summary = self.summary()  # Will use cache if available
        deep_sharpe = summary["deep_hedge"]["sharpe_ratio"]
        bs_sharpe = summary["bs_baseline"]["sharpe_ratio"]

        return (
            f"BacktestResults(n_paths={self.n_paths}, n_steps={self.n_steps}, "
            f"deep_sharpe={deep_sharpe:.3f}, bs_sharpe={bs_sharpe:.3f})"
        )
