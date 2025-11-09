#!/usr/bin/env python3
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def plot_metric_distributions(df: pd.DataFrame, metrics: list, output_dir: Path):
    """Plot distribution of key metrics across all models."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        if metric not in df.columns:
            continue

        ax = axes[idx]
        df[metric].hist(bins=30, ax=ax, edgecolor="black", alpha=0.7)
        ax.set_title(metric.replace("dh_", "").replace("_", " ").title())
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.axvline(df[metric].median(), color="red", linestyle="--", label="Median")
        ax.legend()

    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_file = output_dir / "metric_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_file}")


def plot_performance_by_architecture(df: pd.DataFrame, output_dir: Path):
    """Compare performance across model architectures."""
    metrics = ["dh_sharpe_ratio", "dh_mean", "dh_cvar_95", "dh_win_rate"]
    metrics = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df.boxplot(column=metric, by="model_type", ax=ax)
        ax.set_title(metric.replace("dh_", "").replace("_", " ").title())
        ax.set_xlabel("Model Type")
        ax.set_ylabel("Value")
        plt.sca(ax)
        plt.xticks(rotation=0)

    plt.suptitle("")
    fig.suptitle("Performance by Model Architecture", fontsize=16, y=1.0)
    plt.tight_layout()
    output_file = output_dir / "performance_by_architecture.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_file}")


def plot_performance_by_hyperparameter(
    df: pd.DataFrame, param: str, metric: str, output_dir: Path
):
    """Plot how performance varies with a hyperparameter."""
    if param not in df.columns or metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if df[param].dtype in ["int64", "float64"]:
        grouped = df.groupby(param)[metric].agg(["mean", "std"])
        ax.errorbar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
        )
        ax.set_xlabel(param.replace("_", " ").title())
    else:
        df.boxplot(column=metric, by=param, ax=ax)
        plt.sca(ax)
        plt.xticks(rotation=45)

    ax.set_ylabel(metric.replace("dh_", "").replace("_", " ").title())
    ax.set_title(
        f'{metric.replace("dh_", "").replace("_", " ").title()} by {param.replace("_", " ").title()}'
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"{metric}_by_{param}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_file}")


def plot_heatmap(
    df: pd.DataFrame, param1: str, param2: str, metric: str, output_dir: Path
):
    """Create 2D heatmap of metric values."""
    if param1 not in df.columns or param2 not in df.columns or metric not in df.columns:
        return

    pivot = df.pivot_table(values=metric, index=param1, columns=param2, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=pivot.mean().mean(),
        ax=ax,
        cbar_kws={"label": metric.replace("dh_", "").replace("_", " ").title()},
    )
    ax.set_title(
        f'{metric.replace("dh_", "").replace("_", " ").title()}\nby {param1.replace("_", " ").title()} and {param2.replace("_", " ").title()}'
    )
    ax.set_xlabel(param2.replace("_", " ").title())
    ax.set_ylabel(param1.replace("_", " ").title())

    plt.tight_layout()
    output_file = output_dir / f"heatmap_{param1}_{param2}_{metric}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_file}")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    """Plot correlation between hyperparameters and performance metrics."""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    numeric_cols = [
        c
        for c in numeric_cols
        if not c.startswith("bs_")
        and not c.startswith("test_")
        and not c.startswith("train_")
    ]

    corr_df = df[numeric_cols].copy()

    for col in ["model_type", "risk_measure"]:
        if col in df.columns:
            corr_df[col] = df[col].astype("category").cat.codes

    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Matrix: Hyperparameters and Performance Metrics")

    plt.tight_layout()
    output_file = output_dir / "correlation_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_file}")


def plot_deep_vs_baseline(df: pd.DataFrame, output_dir: Path):
    """Scatter plot comparing deep hedge vs baseline performance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for model_type in df["model_type"].unique():
        subset = df[df["model_type"] == model_type]
        ax.scatter(
            subset["bs_sharpe_ratio"],
            subset["dh_sharpe_ratio"],
            label=model_type.upper(),
            alpha=0.6,
            s=50,
        )

    ax.plot([-4, 0], [-4, 0], "k--", alpha=0.5, label="Equal Performance")
    ax.set_xlabel("Black-Scholes Sharpe Ratio")
    ax.set_ylabel("Deep Hedge Sharpe Ratio")
    ax.set_title("Deep Hedge vs Black-Scholes: Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for model_type in df["model_type"].unique():
        subset = df[df["model_type"] == model_type]
        ax.scatter(
            subset["bs_mean"],
            subset["dh_mean"],
            label=model_type.upper(),
            alpha=0.6,
            s=50,
        )

    min_val, max_val = (
        df[["bs_mean", "dh_mean"]].min().min(),
        df[["bs_mean", "dh_mean"]].max().max(),
    )
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="Equal Performance",
    )
    ax.set_xlabel("Black-Scholes Mean PnL")
    ax.set_ylabel("Deep Hedge Mean PnL")
    ax.set_title("Deep Hedge vs Black-Scholes: Mean PnL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "deep_vs_baseline.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_file}")


def plot_risk_return_tradeoff(df: pd.DataFrame, output_dir: Path):
    """Plot risk-return tradeoff."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for model_type in df["model_type"].unique():
        subset = df[df["model_type"] == model_type]
        ax.scatter(
            subset["dh_std"],
            subset["dh_mean"],
            label=model_type.upper(),
            alpha=0.6,
            s=100,
        )

    ax.set_xlabel("Risk (PnL Std Dev)")
    ax.set_ylabel("Return (Mean PnL)")
    ax.set_title("Risk-Return Tradeoff for Deep Hedge Models")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "risk_return_tradeoff.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_file}")


def plot_top_models_comparison(df: pd.DataFrame, top_n: int, output_dir: Path):
    """Compare top N models across multiple metrics."""
    top_models = df.nlargest(top_n, "dh_sharpe_ratio")

    metrics = ["dh_sharpe_ratio", "dh_mean", "dh_cvar_95", "dh_win_rate"]
    metrics = [m for m in metrics if m in top_models.columns]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(top_models))
    width = 0.2

    for idx, metric in enumerate(metrics):
        values = top_models[metric].values
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
        ax.bar(
            x + idx * width,
            normalized,
            width,
            label=metric.replace("dh_", "").replace("_", " ").title(),
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Normalized Value")
    ax.set_title(f"Top {top_n} Models Comparison (Normalized Metrics)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(
        [
            f"{m['model_type']}-{m['n_layers']}L-{m['n_units']}U"
            for _, m in top_models.iterrows()
        ],
        rotation=45,
        ha="right",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_file = output_dir / f"top_{top_n}_models_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_file}")


def plot_risk_measure_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare entropic vs expected shortfall risk measures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = ["dh_sharpe_ratio", "dh_mean", "dh_cvar_95", "dh_win_rate"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df.boxplot(column=metric, by="risk_measure", ax=ax)
        ax.set_title(metric.replace("dh_", "").replace("_", " ").title())
        ax.set_xlabel("Risk Measure")
        ax.set_ylabel("Value")
        plt.sca(ax)

    plt.suptitle("")
    fig.suptitle("Performance by Risk Measure", fontsize=16, y=1.0)
    plt.tight_layout()
    output_file = output_dir / "risk_measure_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize hyperparameter tuning results"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="hparam_tuning/analysis/all_results.pkl",
        help="Path to aggregated results file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hparam_tuning/analysis/plots",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top models for comparison"
    )
    args = parser.parse_args()

    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results from {results_file}")
    df = pd.read_pickle(results_file)
    logger.info(f"Loaded {len(df)} models")

    logger.info("\nGenerating visualizations...")

    logger.info("\n1. Metric distributions")
    metrics = [
        "dh_sharpe_ratio",
        "dh_mean",
        "dh_cvar_95",
        "dh_sortino_ratio",
        "dh_win_rate",
        "dh_sharpe_improvement",
    ]
    plot_metric_distributions(df, metrics, output_dir)

    logger.info("\n2. Performance by architecture")
    plot_performance_by_architecture(df, output_dir)

    logger.info("\n3. Performance by hyperparameters")
    for param in ["n_layers", "n_units", "risk_measure", "risk_param"]:
        plot_performance_by_hyperparameter(df, param, "dh_sharpe_ratio", output_dir)

    logger.info("\n4. Heatmaps")
    plot_heatmap(df, "n_layers", "n_units", "dh_sharpe_ratio", output_dir)
    plot_heatmap(df, "model_type", "risk_measure", "dh_sharpe_ratio", output_dir)

    logger.info("\n5. Correlation matrix")
    plot_correlation_matrix(df, output_dir)

    logger.info("\n6. Deep hedge vs baseline")
    plot_deep_vs_baseline(df, output_dir)

    logger.info("\n7. Risk-return tradeoff")
    plot_risk_return_tradeoff(df, output_dir)

    logger.info("\n8. Top models comparison")
    plot_top_models_comparison(df, args.top_n, output_dir)

    logger.info("\n9. Risk measure comparison")
    plot_risk_measure_comparison(df, output_dir)

    logger.info(f"\nAll plots saved to {output_dir}/")
    logger.info(f"Generated {len(list(output_dir.glob('*.png')))} visualizations")


if __name__ == "__main__":
    main()
