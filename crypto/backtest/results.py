from typing import Dict, Optional, Any, TYPE_CHECKING
import json
from datetime import datetime
from pathlib import Path
import torch
from torch import Tensor
import numpy as np

if TYPE_CHECKING:
    from .config import BacktestConfig

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

    def __init__(
        self,
        deep_pnl: Tensor,
        bs_pnl: Tensor,
        deep_positions: Tensor,
        bs_positions: Tensor,
        spots: Tensor,
        config: Optional["BacktestConfig"] = None,
    ):
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

        # Bootstrap metadata (NEW)
        self.bootstrap_metadata = self._extract_bootstrap_metadata()

    def _validate_inputs(
        self,
        deep_pnl: Tensor,
        bs_pnl: Tensor,
        deep_positions: Tensor,
        bs_positions: Tensor,
        spots: Tensor,
    ) -> None:
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

    def _extract_bootstrap_metadata(self) -> Dict[str, Any]:
        metadata = {}

        if self.config is None:
            return metadata

        # Basic bootstrap configuration
        metadata["mode"] = self.config.bootstrap_mode
        metadata["n_paths"] = self.n_paths
        metadata["n_steps"] = self.n_steps

        # Strike and moneyness info
        if hasattr(self.config, "strike"):
            metadata["strike"] = self.config.strike

        if hasattr(self.config, "initial_spot"):
            metadata["initial_spot_config"] = self.config.initial_spot

        if hasattr(self.config, "effective_target_moneyness"):
            try:
                target_moneyness = self.config.effective_target_moneyness
                if target_moneyness is not None:
                    metadata["target_moneyness"] = target_moneyness
            except Exception:
                pass  # Skip if calculation fails

        # If normalize_spot mode, add rescaling stats
        if self.config.bootstrap_mode == "normalize_spot":
            if hasattr(self.config, "strike") and hasattr(
                self.config, "effective_target_moneyness"
            ):
                try:
                    target_moneyness = self.config.effective_target_moneyness
                    if target_moneyness is not None:
                        target_initial_spot = self.config.strike * target_moneyness
                        metadata["target_initial_spot"] = target_initial_spot

                        # Actual initial spots from paths
                        actual_initial_spots = self.spots[:, 0]
                        metadata["actual_initial_spots"] = {
                            "min": float(actual_initial_spots.min().item()),
                            "max": float(actual_initial_spots.max().item()),
                            "mean": float(actual_initial_spots.mean().item()),
                            "std": float(actual_initial_spots.std().item()),
                        }

                        # Scale factors if available (stored in spots tensor attributes)
                        if hasattr(self.spots, "_bootstrap_scale_factors"):
                            scale_factors = self.spots._bootstrap_scale_factors
                            metadata["scale_factors"] = {
                                "min": float(scale_factors.min().item()),
                                "max": float(scale_factors.max().item()),
                                "mean": float(scale_factors.mean().item()),
                                "std": float(scale_factors.std().item()),
                            }
                except Exception:
                    pass  # Skip if calculation fails

        # If absolute_strike mode, add moneyness distribution
        elif self.config.bootstrap_mode == "absolute_strike":
            if hasattr(self.config, "strike"):
                try:
                    initial_moneyness = self.spots[:, 0] / self.config.strike
                    log_moneyness = torch.log(initial_moneyness)

                    metadata["initial_log_moneyness"] = {
                        "p5": float(torch.quantile(log_moneyness, 0.05).item()),
                        "p50": float(torch.quantile(log_moneyness, 0.50).item()),
                        "p95": float(torch.quantile(log_moneyness, 0.95).item()),
                        "mean": float(log_moneyness.mean().item()),
                        "std": float(log_moneyness.std().item()),
                    }
                except Exception:
                    pass  # Skip if calculation fails

        return metadata

    def _get_time_axis(self, time_unit: str = "steps") -> tuple:
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
        data = {
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "summary": self.summary(),
            "bootstrap": self.bootstrap_metadata,  # NEW: Bootstrap metadata
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

    def compare_strategies(self) -> dict:
        summary = self.summary()
        deep = summary["deep_hedge"]
        bs = summary["bs_baseline"]

        # Epsilon for tie-handling (avoid flapping on numerically equal metrics)
        EPSILON = 1e-9

        # Determine winners with epsilon tolerance
        # Higher is better for: mean PnL, CVaR (less negative loss)
        # Lower is better for: max drawdown (smaller loss)
        winners = {}

        # Mean PnL
        if abs(deep["mean"] - bs["mean"]) < EPSILON:
            winners["mean"] = "tie"
        else:
            winners["mean"] = (
                "deep_hedge" if deep["mean"] > bs["mean"] else "bs_baseline"
            )

        # Sharpe ratio: neutralize if BOTH are negative (both losing)
        if deep["sharpe_ratio"] < 0 and bs["sharpe_ratio"] < 0:
            winners["sharpe_ratio"] = "tie"  # Don't award win for "less negative"
        elif abs(deep["sharpe_ratio"] - bs["sharpe_ratio"]) < EPSILON:
            winners["sharpe_ratio"] = "tie"
        else:
            winners["sharpe_ratio"] = (
                "deep_hedge"
                if deep["sharpe_ratio"] > bs["sharpe_ratio"]
                else "bs_baseline"
            )

        # CVaR: less negative (closer to zero) is better, so higher value wins
        if abs(deep["cvar_95"] - bs["cvar_95"]) < EPSILON:
            winners["cvar_95"] = "tie"
        else:
            winners["cvar_95"] = (
                "deep_hedge" if deep["cvar_95"] > bs["cvar_95"] else "bs_baseline"
            )

        # Max DD: positive value = loss, so LOWER is better
        if abs(deep["max_drawdown"] - bs["max_drawdown"]) < EPSILON:
            winners["max_drawdown"] = "tie"
        else:
            winners["max_drawdown"] = (
                "deep_hedge"
                if deep["max_drawdown"] < bs["max_drawdown"]
                else "bs_baseline"
            )

        # Calculate differences
        differences = {
            "mean": deep["mean"] - bs["mean"],
            "std": deep["std"] - bs["std"],
            "sharpe_ratio": deep["sharpe_ratio"] - bs["sharpe_ratio"],
            "sortino_ratio": deep["sortino_ratio"] - bs["sortino_ratio"],
            "cvar_95": deep["cvar_95"] - bs["cvar_95"],
            "max_drawdown": deep["max_drawdown"] - bs["max_drawdown"],
            "win_rate": deep["win_rate"] - bs["win_rate"],
        }

        # Calculate reductions for metrics where lower is better (use absolute values)
        # This gives cleaner "X% reduction in volatility" messaging
        reductions = {}
        if bs["std"] != 0:
            reductions["std"] = (
                (abs(differences["std"]) / bs["std"]) * 100
                if differences["std"] < 0
                else 0.0
            )
        else:
            reductions["std"] = 0.0

        if bs["max_drawdown"] != 0:
            reductions["max_drawdown"] = (
                (abs(differences["max_drawdown"]) / bs["max_drawdown"]) * 100
                if differences["max_drawdown"] < 0
                else 0.0
            )
        else:
            reductions["max_drawdown"] = 0.0

        # Calculate percentage changes (only for appropriate metrics - backward compatibility)
        # Following reviewer guidance: restrict to dollar metrics (mean) and win_rate
        # Avoid misleading percentage changes on ratios (Sharpe/Sortino)
        percentage_changes = {}

        # Mean PnL (dollar metric - appropriate)
        if bs["mean"] != 0:
            percentage_changes["mean"] = (differences["mean"] / abs(bs["mean"])) * 100
        else:
            percentage_changes["mean"] = 0.0

        # Win rate (percentage metric - appropriate)
        if bs["win_rate"] != 0:
            percentage_changes["win_rate"] = (
                differences["win_rate"] / bs["win_rate"]
            ) * 100
        else:
            percentage_changes["win_rate"] = 0.0

        # Std: include for backward compat, but prefer using reductions instead
        if bs["std"] != 0:
            percentage_changes["std"] = (differences["std"] / bs["std"]) * 100
        else:
            percentage_changes["std"] = 0.0

        # Sharpe/Sortino: include for backward compat, but these are often misleading
        if bs["sharpe_ratio"] != 0:
            percentage_changes["sharpe_ratio"] = (
                differences["sharpe_ratio"] / abs(bs["sharpe_ratio"])
            ) * 100
        else:
            percentage_changes["sharpe_ratio"] = 0.0

        # Count wins on key metrics (ties don't count for either side)
        key_metrics = ["mean", "sharpe_ratio", "cvar_95", "max_drawdown"]
        deep_wins = sum(1 for metric in key_metrics if winners[metric] == "deep_hedge")
        ties = sum(1 for metric in key_metrics if winners[metric] == "tie")

        # Overall assessment
        if deep_wins >= 3:
            assessment = "superior"
        elif deep_wins >= 2:
            assessment = "mixed"
        else:
            assessment = "underperformed"

        return {
            "winners": winners,
            "differences": differences,
            "percentage_changes": percentage_changes,  # For backward compatibility
            "reductions": reductions,  # NEW: cleaner messaging for "lower is better" metrics
            "summary": {
                "deep_wins": deep_wins,
                "ties": ties,
                "total_metrics": len(key_metrics),
                "assessment": assessment,
            },
        }

    def print_summary(self, detailed: bool = True, emoji: bool = True) -> None:
        summary = self.summary()
        deep = summary["deep_hedge"]
        bs = summary["bs_baseline"]

        print("\n" + "=" * 60)
        title = "📊 BACKTEST RESULTS SUMMARY" if emoji else "BACKTEST RESULTS SUMMARY"
        print(title)
        print("=" * 60)

        if detailed:
            # Full detailed view (grouped by category)
            self._print_strategy_details("Deep Hedge", deep, emoji=emoji)
            self._print_strategy_details("Black-Scholes", bs, emoji=emoji)
            self._print_comparison(deep, bs)
        else:
            # Compact view (key metrics only)
            self._print_compact_comparison(deep, bs)

        print("=" * 60)

    def _print_strategy_details(
        self, name: str, metrics: dict, emoji: bool = True
    ) -> None:
        print(f"\n{name.upper()}")
        print("─" * 60)

        profit_label = "💰 Profitability:" if emoji else "Profitability:"
        print(f"\n{profit_label}")
        print(f"   Mean PnL:        ${metrics['mean']:>10.2f}")
        print(f"   Std Dev:         ${metrics['std']:>10.2f}")
        print(f"   Min PnL:         ${metrics['min']:>10.2f}")
        print(f"   Max PnL:         ${metrics['max']:>10.2f}")
        print(f"   Median PnL:      ${metrics['median']:>10.2f}")

        risk_adj_label = "📈 Risk-Adjusted:" if emoji else "Risk-Adjusted:"
        print(f"\n{risk_adj_label}")
        print(f"   Sharpe Ratio:    {metrics['sharpe_ratio']:>10.3f}")
        print(f"   Sortino Ratio:   {metrics['sortino_ratio']:>10.3f}")

        risk_label = "⚠️  Risk Metrics:" if emoji else "Risk Metrics:"
        print(f"\n{risk_label}")
        print(f"   CVaR (95%):      ${metrics['cvar_95']:>10.2f}")
        print(f"   Max Drawdown:    ${metrics['max_drawdown']:>10.2f}")
        print(f"   Calmar Ratio:    {metrics['calmar_ratio']:>10.3f}")
        print(f"   Win Rate:        {metrics['win_rate']:>9.1%}")

    def _print_comparison(self, deep: dict, bs: dict) -> None:
        print(f"\nCOMPARISON (Deep Hedge vs Black-Scholes)")
        print("─" * 60)

        mean_diff = deep["mean"] - bs["mean"]
        mean_pct = (mean_diff / abs(bs["mean"]) * 100) if bs["mean"] != 0 else 0

        print(f"\n   Mean PnL:        ${mean_diff:>10.2f} ({mean_pct:+.1f}%)")
        print(f"   Sharpe Ratio:    {deep['sharpe_ratio'] - bs['sharpe_ratio']:>10.3f}")
        print(f"   CVaR (95%):      ${deep['cvar_95'] - bs['cvar_95']:>10.2f}")
        print(
            f"   Max Drawdown:    ${deep['max_drawdown'] - bs['max_drawdown']:>10.2f}"
        )

    def _print_compact_comparison(self, deep: dict, bs: dict) -> None:
        print(
            f"\nStrategy          Mean PnL    Std Dev    Sharpe    CVaR 95%    Win Rate"
        )
        print("─" * 78)
        print(
            f"Deep Hedge     {deep['mean']:>10.2f}  {deep['std']:>9.2f}  {deep['sharpe_ratio']:>7.3f}  {deep['cvar_95']:>10.2f}  {deep['win_rate']:>8.1%}"
        )
        print(
            f"Black-Scholes  {bs['mean']:>10.2f}  {bs['std']:>9.2f}  {bs['sharpe_ratio']:>7.3f}  {bs['cvar_95']:>10.2f}  {bs['win_rate']:>8.1%}"
        )

        mean_diff = deep["mean"] - bs["mean"]
        print("─" * 78)
        print(
            f"Difference     {mean_diff:>10.2f}  {deep['std'] - bs['std']:>9.2f}  {deep['sharpe_ratio'] - bs['sharpe_ratio']:>7.3f}  {deep['cvar_95'] - bs['cvar_95']:>10.2f}  {(deep['win_rate'] - bs['win_rate'])*100:>7.1f}%"
        )

    def print_key_insights(self, emoji: bool = True) -> None:
        comparison = self.compare_strategies()
        summary = self.summary()
        deep = summary["deep_hedge"]
        bs = summary["bs_baseline"]

        print("\n" + "=" * 60)
        title = "💡 KEY INSIGHTS" if emoji else "KEY INSIGHTS"
        print(title)
        print("=" * 60)

        # Show winners on each key metric
        winners = comparison["winners"]

        # Format winner lines with appropriate verb (wins/ties)
        winner_verb = lambda w: "wins" if w != "tie" else "(tie)"

        print(
            f"\n1. Mean PnL:      {self._format_winner(winners['mean'])} {winner_verb(winners['mean'])}"
        )

        # For Sharpe: annotate if neutralized due to both being negative
        sharpe_annotation = ""
        if (
            winners["sharpe_ratio"] == "tie"
            and deep["sharpe_ratio"] < 0
            and bs["sharpe_ratio"] < 0
        ):
            sharpe_annotation = " (both negative)"
        winner_text = (
            "wins (risk-adjusted)" if winners["sharpe_ratio"] != "tie" else "(tie)"
        )
        print(
            f"2. Sharpe Ratio:  {self._format_winner(winners['sharpe_ratio'])} {winner_text}{sharpe_annotation}"
        )

        cvar_text = "has better tail risk" if winners["cvar_95"] != "tie" else "(tie)"
        print(
            f"3. CVaR (95%):    {self._format_winner(winners['cvar_95'])} {cvar_text}"
        )

        dd_text = (
            "has better drawdown protection"
            if winners["max_drawdown"] != "tie"
            else "(tie)"
        )
        print(
            f"4. Max Drawdown:  {self._format_winner(winners['max_drawdown'])} {dd_text}"
        )

        # Overall assessment
        deep_wins = comparison["summary"]["deep_wins"]
        ties = comparison["summary"]["ties"]
        total = comparison["summary"]["total_metrics"]
        assessment = comparison["summary"]["assessment"]

        # Show wins and ties
        overall_icon = "📊 " if emoji else ""
        if ties > 0:
            print(
                f"\n{overall_icon}Overall: Deep Hedge wins on {deep_wins}/{total} key metrics ({ties} ties)"
            )
        else:
            print(
                f"\n{overall_icon}Overall: Deep Hedge wins on {deep_wins}/{total} key metrics"
            )

        if assessment == "superior":
            conclusion_icon = "✅ " if emoji else ""
            print(
                f"\n{conclusion_icon}CONCLUSION: Deep Hedge demonstrates superior performance"
            )
        elif assessment == "mixed":
            conclusion_icon = "⚖️  " if emoji else ""
            print(
                f"\n{conclusion_icon}CONCLUSION: Mixed results - context-dependent decision"
            )
        else:
            conclusion_icon = "⚠️  " if emoji else ""
            print(
                f"\n{conclusion_icon}CONCLUSION: Black-Scholes performed better in this scenario"
            )

        # Highlight most important differences
        diffs = comparison["differences"]
        reductions = comparison["reductions"]

        improve_icon = "🔑 " if emoji else ""
        print(f"\n{improve_icon}Key Improvements (Deep Hedge vs BS):")

        bullet = "•" if emoji else "-"

        # Mean PnL: compute percentage change on the fly (appropriate for dollar metrics)
        if diffs["mean"] > 0:
            if deep["mean"] < 0 and bs["mean"] < 0:
                # Both are losses - use "lower loss" language
                pct_mean = (
                    (abs(diffs["mean"]) / abs(bs["mean"]) * 100)
                    if bs["mean"] != 0
                    else 0
                )
                print(f"   {bullet} {pct_mean:.1f}% lower loss (mean PnL)")
            else:
                pct_mean = (
                    (abs(diffs["mean"]) / abs(bs["mean"]) * 100)
                    if bs["mean"] != 0
                    else 0
                )
                print(f"   {bullet} {pct_mean:.1f}% better mean PnL")

        # Volatility: use reductions dict for cleaner messaging
        if reductions["std"] > 0:
            print(f"   {bullet} {reductions['std']:.1f}% reduction in volatility")

        # CVaR: use dollar difference (not percentage)
        if diffs["cvar_95"] > 0:
            print(
                f"   {bullet} ${abs(diffs['cvar_95']):.2f} better CVaR (reduced tail risk)"
            )

        # Max drawdown: use reductions dict
        if reductions["max_drawdown"] > 0:
            print(
                f"   {bullet} {reductions['max_drawdown']:.1f}% reduction in max drawdown"
            )

        print("=" * 60)

    def _format_winner(self, strategy: str) -> str:
        if strategy == "tie":
            return "Tie"
        return "Deep Hedge" if strategy == "deep_hedge" else "Black-Scholes"

    def __repr__(self) -> str:
        summary = self.summary()  # Will use cache if available
        deep_sharpe = summary["deep_hedge"]["sharpe_ratio"]
        bs_sharpe = summary["bs_baseline"]["sharpe_ratio"]

        return (
            f"BacktestResults(n_paths={self.n_paths}, n_steps={self.n_steps}, "
            f"deep_sharpe={deep_sharpe:.3f}, bs_sharpe={bs_sharpe:.3f})"
        )
