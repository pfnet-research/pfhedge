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

    def _get_time_axis(self, time_unit: str = "steps") -> tuple:
        """Get time axis values and label based on config.

        Args:
            time_unit: One of 'auto', 'steps', 'hours', 'days' (default: 'steps')
                - 'auto': Automatically choose based on dt_hours from config
                    - If dt >= 24 hours: use days
                    - If dt >= 1 hour: use hours
                    - Otherwise: use steps
                - 'steps': Use time step indices
                - 'hours': Convert to hours using config.dt
                - 'days': Convert to days using config.dt

        Returns:
            Tuple of (time_values, xlabel) where:
            - time_values: np.ndarray of time axis values
            - xlabel: str for axis label

        Examples:
            >>> # Auto mode selects best unit based on dt
            >>> time_vals, label = results._get_time_axis("auto")
            >>>
            >>> # Explicit unit selection
            >>> time_vals, label = results._get_time_axis("days")
            >>> ax.plot(time_vals, data)
            >>> ax.set_xlabel(label)
        """
        time_steps = np.arange(self.n_steps)

        # Auto mode: choose best unit based on dt_hours
        if time_unit == "auto":
            if self.config and hasattr(self.config, "dt_hours"):
                dt_hours = self.config.dt_hours
                if dt_hours >= 24:
                    time_unit = "days"
                elif dt_hours >= 1:
                    time_unit = "hours"
                else:
                    time_unit = "steps"
            else:
                # No config available, fall back to steps
                time_unit = "steps"

        if time_unit == "steps":
            return time_steps, "Time Step"

        # For hours/days, we need dt from config
        if self.config is None or not hasattr(self.config, "dt"):
            # Fallback to steps if no config available
            return time_steps, "Time Step"

        dt_years = self.config.dt
        hours_per_year = 24 * 365.25

        if time_unit == "hours":
            time_values = time_steps * dt_years * hours_per_year
            return time_values, "Time (hours)"
        elif time_unit == "days":
            time_values = time_steps * dt_years * (hours_per_year / 24)
            return time_values, "Time (days)"
        else:
            raise ValueError(
                f"Invalid time_unit: {time_unit}. Use 'auto', 'steps', 'hours', or 'days'"
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

    def plot_pnl_comparison(
        self,
        path_indices: Optional[list] = None,
        show_mean: bool = True,
        time_unit: str = "steps",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ):
        """Plot cumulative PnL comparison between deep hedge and BS baseline.

        Args:
            path_indices: List of path indices to plot. If None, plots mean only.
            show_mean: If True, show mean PnL across all paths (default: True)
            time_unit: Time axis unit - 'steps', 'hours', or 'days' (default: 'steps')
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save figure. If None, displays plot.

        Returns:
            matplotlib Figure object

        Examples:
            >>> # Plot mean PnL only
            >>> results.plot_pnl_comparison()
            >>>
            >>> # Plot first 5 paths plus mean
            >>> results.plot_pnl_comparison(path_indices=[0, 1, 2, 3, 4])
            >>>
            >>> # Plot with time in days
            >>> results.plot_pnl_comparison(time_unit="days")
            >>>
            >>> # Save to file
            >>> results.plot_pnl_comparison(save_path="pnl_comparison.png")
        """
        import matplotlib.pyplot as plt

        # Validate and limit path_indices
        if path_indices is not None and len(path_indices) > 50:
            print(
                f"⚠️  Warning: Plotting {len(path_indices)} paths may be slow. Consider using fewer paths."
            )

        fig, ax = plt.subplots(figsize=figsize)

        # Get time axis
        time_values, time_label = self._get_time_axis(time_unit)

        # Plot individual paths if requested
        if path_indices is not None:
            for idx in path_indices:
                if idx >= self.n_paths:
                    continue
                ax.plot(
                    time_values,
                    self.deep_pnl[idx].cpu().numpy(),
                    color="blue",
                    alpha=0.3,
                    linewidth=0.5,
                )
                ax.plot(
                    time_values,
                    self.bs_pnl[idx].cpu().numpy(),
                    color="orange",
                    alpha=0.3,
                    linewidth=0.5,
                )

        # Plot mean PnL
        if show_mean:
            deep_mean = self.deep_pnl.mean(dim=0).cpu().numpy()
            bs_mean = self.bs_pnl.mean(dim=0).cpu().numpy()

            ax.plot(
                time_values,
                deep_mean,
                color="blue",
                linewidth=2,
                label="Deep Hedge (mean)",
            )
            ax.plot(
                time_values,
                bs_mean,
                color="orange",
                linewidth=2,
                label="BS Baseline (mean)",
            )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(time_label)
        ax.set_ylabel("Cumulative PnL ($)")
        ax.set_title("Cumulative PnL Comparison: Deep Hedge vs Black-Scholes")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved PnL comparison plot to: {save_path}")

        return fig

    def plot_pnl_distribution(
        self, bins: int = 50, figsize: tuple = (12, 6), save_path: Optional[str] = None
    ):
        """Plot final PnL distribution for both strategies.

        Args:
            bins: Number of histogram bins (default: 50)
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save figure. If None, displays plot.

        Returns:
            matplotlib Figure object

        Examples:
            >>> results.plot_pnl_distribution()
            >>> results.plot_pnl_distribution(bins=30, save_path="pnl_dist.png")
        """
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Get final PnL
        deep_final = self.deep_pnl[:, -1].cpu().numpy()
        bs_final = self.bs_pnl[:, -1].cpu().numpy()

        # Plot deep hedge distribution
        ax1.hist(deep_final, bins=bins, alpha=0.7, color="blue", edgecolor="black")
        ax1.axvline(
            deep_final.mean(), color="red", linestyle="--", linewidth=2, label="Mean"
        )
        ax1.axvline(
            np.median(deep_final),
            color="green",
            linestyle="--",
            linewidth=2,
            label="Median",
        )
        ax1.set_xlabel("Final PnL ($)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Deep Hedge: Final PnL Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot BS baseline distribution
        ax2.hist(bs_final, bins=bins, alpha=0.7, color="orange", edgecolor="black")
        ax2.axvline(
            bs_final.mean(), color="red", linestyle="--", linewidth=2, label="Mean"
        )
        ax2.axvline(
            np.median(bs_final),
            color="green",
            linestyle="--",
            linewidth=2,
            label="Median",
        )
        ax2.set_xlabel("Final PnL ($)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Black-Scholes: Final PnL Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved PnL distribution plot to: {save_path}")

        return fig

    def plot_positions(
        self,
        path_indices: Optional[list] = None,
        show_mean: bool = True,
        time_unit: str = "steps",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ):
        """Plot hedge positions over time for both strategies.

        Args:
            path_indices: List of path indices to plot. If None, plots mean only.
            show_mean: If True, show mean positions across all paths (default: True)
            time_unit: Time axis unit - 'steps', 'hours', or 'days' (default: 'steps')
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save figure. If None, displays plot.

        Returns:
            matplotlib Figure object

        Examples:
            >>> results.plot_positions()
            >>> results.plot_positions(path_indices=[0, 1, 2])
            >>> results.plot_positions(time_unit="days")
            >>> results.plot_positions(save_path="positions.png")
        """
        import matplotlib.pyplot as plt

        # Validate and limit path_indices
        if path_indices is not None and len(path_indices) > 50:
            print(
                f"⚠️  Warning: Plotting {len(path_indices)} paths may be slow. Consider using fewer paths."
            )

        fig, ax = plt.subplots(figsize=figsize)

        # Get time axis
        time_values, time_label = self._get_time_axis(time_unit)

        # Plot individual paths if requested
        if path_indices is not None:
            for idx in path_indices:
                if idx >= self.n_paths:
                    continue
                ax.plot(
                    time_values,
                    self.deep_positions[idx].cpu().numpy(),
                    color="blue",
                    alpha=0.3,
                    linewidth=0.5,
                )
                ax.plot(
                    time_values,
                    self.bs_positions[idx].cpu().numpy(),
                    color="orange",
                    alpha=0.3,
                    linewidth=0.5,
                )

        # Plot mean positions
        if show_mean:
            deep_mean = self.deep_positions.mean(dim=0).cpu().numpy()
            bs_mean = self.bs_positions.mean(dim=0).cpu().numpy()

            ax.plot(
                time_values,
                deep_mean,
                color="blue",
                linewidth=2,
                label="Deep Hedge (mean)",
            )
            ax.plot(
                time_values,
                bs_mean,
                color="orange",
                linewidth=2,
                label="BS Delta (mean)",
            )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(time_label)
        ax.set_ylabel("Hedge Position (units)")
        ax.set_title("Hedge Positions Over Time: Deep Hedge vs Black-Scholes Delta")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved positions plot to: {save_path}")

        return fig

    def plot_all(
        self,
        path_indices: Optional[list] = None,
        time_unit: str = "steps",
        figsize: tuple = (16, 12),
        save_path: Optional[str] = None,
    ):
        """Create comprehensive visualization with all plots.

        Creates a 2x2 grid with:
        - Top left: PnL comparison
        - Top right: PnL distribution
        - Bottom left: Hedge positions
        - Bottom right: Performance metrics table

        Args:
            path_indices: List of path indices to plot in line plots
            time_unit: Time axis unit - 'steps', 'hours', or 'days' (default: 'steps')
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save figure. If None, displays plot.

        Returns:
            matplotlib Figure object

        Examples:
            >>> results.plot_all()
            >>> results.plot_all(path_indices=[0, 1, 2], save_path="backtest_summary.png")
            >>> results.plot_all(time_unit="days")
        """
        import matplotlib.pyplot as plt

        # Validate and limit path_indices
        if path_indices is not None and len(path_indices) > 50:
            print(
                f"⚠️  Warning: Plotting {len(path_indices)} paths may be slow. Consider using fewer paths."
            )

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Get time axis
        time_values, time_label = self._get_time_axis(time_unit)

        # Top left: PnL comparison
        ax1 = fig.add_subplot(gs[0, 0])
        if path_indices is not None:
            for idx in path_indices:
                if idx >= self.n_paths:
                    continue
                ax1.plot(
                    time_values,
                    self.deep_pnl[idx].cpu().numpy(),
                    color="blue",
                    alpha=0.3,
                    linewidth=0.5,
                )
                ax1.plot(
                    time_values,
                    self.bs_pnl[idx].cpu().numpy(),
                    color="orange",
                    alpha=0.3,
                    linewidth=0.5,
                )

        deep_mean = self.deep_pnl.mean(dim=0).cpu().numpy()
        bs_mean = self.bs_pnl.mean(dim=0).cpu().numpy()
        ax1.plot(time_values, deep_mean, color="blue", linewidth=2, label="Deep Hedge")
        ax1.plot(time_values, bs_mean, color="orange", linewidth=2, label="BS Baseline")
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax1.set_xlabel(time_label)
        ax1.set_ylabel("Cumulative PnL ($)")
        ax1.set_title("Cumulative PnL Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Top right: PnL distribution (overlay)
        ax2 = fig.add_subplot(gs[0, 1])
        deep_final = self.deep_pnl[:, -1].cpu().numpy()
        bs_final = self.bs_pnl[:, -1].cpu().numpy()
        ax2.hist(
            deep_final,
            bins=30,
            alpha=0.5,
            color="blue",
            label="Deep Hedge",
            edgecolor="black",
        )
        ax2.hist(
            bs_final,
            bins=30,
            alpha=0.5,
            color="orange",
            label="BS Baseline",
            edgecolor="black",
        )
        ax2.set_xlabel("Final PnL ($)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Final PnL Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Bottom left: Hedge positions
        ax3 = fig.add_subplot(gs[1, 0])
        if path_indices is not None:
            for idx in path_indices:
                if idx >= self.n_paths:
                    continue
                ax3.plot(
                    time_values,
                    self.deep_positions[idx].cpu().numpy(),
                    color="blue",
                    alpha=0.3,
                    linewidth=0.5,
                )
                ax3.plot(
                    time_values,
                    self.bs_positions[idx].cpu().numpy(),
                    color="orange",
                    alpha=0.3,
                    linewidth=0.5,
                )

        deep_pos_mean = self.deep_positions.mean(dim=0).cpu().numpy()
        bs_pos_mean = self.bs_positions.mean(dim=0).cpu().numpy()
        ax3.plot(
            time_values, deep_pos_mean, color="blue", linewidth=2, label="Deep Hedge"
        )
        ax3.plot(
            time_values, bs_pos_mean, color="orange", linewidth=2, label="BS Delta"
        )
        ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax3.set_xlabel(time_label)
        ax3.set_ylabel("Position (units)")
        ax3.set_title("Hedge Positions")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Bottom right: Performance metrics table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        summary = self.summary()
        deep = summary["deep_hedge"]
        bs = summary["bs_baseline"]

        metrics_data = [
            ["Metric", "Deep Hedge", "BS Baseline", "Difference"],
            [
                "Mean PnL",
                f"${deep['mean']:.2f}",
                f"${bs['mean']:.2f}",
                f"${deep['mean']-bs['mean']:+.2f}",
            ],
            [
                "Std PnL",
                f"${deep['std']:.2f}",
                f"${bs['std']:.2f}",
                f"${deep['std']-bs['std']:+.2f}",
            ],
            [
                "Sharpe",
                f"{deep['sharpe_ratio']:.3f}",
                f"{bs['sharpe_ratio']:.3f}",
                f"{deep['sharpe_ratio']-bs['sharpe_ratio']:+.3f}",
            ],
            [
                "Sortino",
                f"{deep['sortino_ratio']:.3f}",
                f"{bs['sortino_ratio']:.3f}",
                f"{deep['sortino_ratio']-bs['sortino_ratio']:+.3f}",
            ],
            [
                "CVaR 95%",
                f"${deep['cvar_95']:.2f}",
                f"${bs['cvar_95']:.2f}",
                f"${deep['cvar_95']-bs['cvar_95']:+.2f}",
            ],
            [
                "Max DD",
                f"${deep['max_drawdown']:.2f}",
                f"${bs['max_drawdown']:.2f}",
                f"${deep['max_drawdown']-bs['max_drawdown']:+.2f}",
            ],
            [
                "Win Rate",
                f"{deep['win_rate']:.1%}",
                f"{bs['win_rate']:.1%}",
                f"{(deep['win_rate']-bs['win_rate'])*100:+.1f}%",
            ],
        ]

        table = ax4.table(
            cellText=metrics_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.3, 0.25, 0.25, 0.2],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax4.set_title("Performance Metrics", fontsize=12, fontweight="bold", pad=20)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved comprehensive plot to: {save_path}")

        return fig

    def generate_report(
        self, filepath: str, include_plots: bool = True, plot_dir: Optional[str] = None
    ) -> dict:
        """Generate a markdown report with backtest results.

        Args:
            filepath: Path to save markdown report (.md file)
            include_plots: If True, generates and embeds plot images
            plot_dir: Directory to save plots. If None, uses same dir as report.

        Returns:
            Dictionary containing:
            - 'report_path': Path to generated report file
            - 'plot_paths': List of generated plot file paths (if include_plots=True)

        Examples:
            >>> result = results.generate_report("backtest_report.md")
            >>> print(result['report_path'])
            >>> print(result['plot_paths'])
            >>>
            >>> results.generate_report("report.md", include_plots=True, plot_dir="plots/")
        """
        from pathlib import Path

        # Determine plot directory
        report_path = Path(filepath)
        if plot_dir is None:
            plot_dir = str(report_path.parent / "plots")

        if include_plots:
            Path(plot_dir).mkdir(parents=True, exist_ok=True)

        # Track generated plot paths
        generated_plots = []

        # Get summary stats
        summary = self.summary()
        deep = summary["deep_hedge"]
        bs = summary["bs_baseline"]

        # Build markdown report
        lines = []
        lines.append("# Backtest Report\n")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Paths:** {self.n_paths:,} | **Steps:** {self.n_steps}\n")

        # Configuration
        if self.config is not None:
            lines.append("\n## Configuration\n")
            if hasattr(self.config, "to_dict"):
                config_dict = self.config.to_dict()
                for key, value in config_dict.items():
                    lines.append(f"- **{key}**: {value}\n")

        # Performance Summary
        lines.append("\n## Performance Summary\n")

        lines.append("\n### Deep Hedge\n")
        lines.append(f"- **Mean PnL**: ${deep['mean']:,.2f}\n")
        lines.append(f"- **Std PnL**: ${deep['std']:,.2f}\n")
        lines.append(f"- **Sharpe Ratio**: {deep['sharpe_ratio']:.3f}\n")
        lines.append(f"- **Sortino Ratio**: {deep['sortino_ratio']:.3f}\n")
        lines.append(f"- **CVaR (95%)**: ${deep['cvar_95']:,.2f}\n")
        lines.append(f"- **Max Drawdown**: ${deep['max_drawdown']:,.2f}\n")
        lines.append(f"- **Win Rate**: {deep['win_rate']:.1%}\n")

        lines.append("\n### Black-Scholes Baseline\n")
        lines.append(f"- **Mean PnL**: ${bs['mean']:,.2f}\n")
        lines.append(f"- **Std PnL**: ${bs['std']:,.2f}\n")
        lines.append(f"- **Sharpe Ratio**: {bs['sharpe_ratio']:.3f}\n")
        lines.append(f"- **Sortino Ratio**: {bs['sortino_ratio']:.3f}\n")
        lines.append(f"- **CVaR (95%)**: ${bs['cvar_95']:,.2f}\n")
        lines.append(f"- **Max Drawdown**: ${bs['max_drawdown']:,.2f}\n")
        lines.append(f"- **Win Rate**: {bs['win_rate']:.1%}\n")

        # Comparison
        lines.append("\n### Comparison (Deep - BS)\n")
        lines.append(f"- **Mean PnL Diff**: ${deep['mean'] - bs['mean']:+,.2f}\n")
        lines.append(
            f"- **Sharpe Diff**: {deep['sharpe_ratio'] - bs['sharpe_ratio']:+.3f}\n"
        )
        lines.append(f"- **CVaR Diff**: ${deep['cvar_95'] - bs['cvar_95']:+,.2f}\n")

        # Include plots if requested
        if include_plots:
            lines.append("\n## Visualizations\n")

            # Save plots (absolute paths for saving)
            plot_dir_path = Path(plot_dir)
            pnl_plot = str(plot_dir_path / "pnl_comparison.png")
            dist_plot = str(plot_dir_path / "pnl_distribution.png")
            pos_plot = str(plot_dir_path / "positions.png")
            all_plot = str(plot_dir_path / "summary.png")

            import matplotlib.pyplot as plt

            self.plot_pnl_comparison(save_path=pnl_plot)
            plt.close()
            generated_plots.append(pnl_plot)

            self.plot_pnl_distribution(save_path=dist_plot)
            plt.close()
            generated_plots.append(dist_plot)

            self.plot_positions(save_path=pos_plot)
            plt.close()
            generated_plots.append(pos_plot)

            self.plot_all(save_path=all_plot)
            plt.close()
            generated_plots.append(all_plot)

            # Create relative paths for markdown links
            # Make paths relative to the report file location
            try:
                pnl_plot_rel = Path(pnl_plot).relative_to(report_path.parent)
                dist_plot_rel = Path(dist_plot).relative_to(report_path.parent)
                pos_plot_rel = Path(pos_plot).relative_to(report_path.parent)
                all_plot_rel = Path(all_plot).relative_to(report_path.parent)
            except ValueError:
                # If relative path fails, use absolute paths
                pnl_plot_rel = pnl_plot
                dist_plot_rel = dist_plot
                pos_plot_rel = pos_plot
                all_plot_rel = all_plot

            # Embed plots in markdown using relative paths
            lines.append("\n### PnL Comparison\n")
            lines.append(f"![PnL Comparison]({pnl_plot_rel})\n")
            lines.append("\n### PnL Distribution\n")
            lines.append(f"![PnL Distribution]({dist_plot_rel})\n")
            lines.append("\n### Hedge Positions\n")
            lines.append(f"![Positions]({pos_plot_rel})\n")
            lines.append("\n### Summary\n")
            lines.append(f"![Summary]({all_plot_rel})\n")

        # Write report
        with open(filepath, "w") as f:
            f.writelines(lines)

        print(f"✅ Generated report: {filepath}")
        if include_plots:
            print(f"   Plots saved to: {plot_dir}/")

        return {
            "report_path": str(filepath),
            "plot_paths": generated_plots,
        }

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
