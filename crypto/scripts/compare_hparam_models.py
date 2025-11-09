#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def rank_models(df: pd.DataFrame, metric: str, ascending: bool = False) -> pd.DataFrame:
    """Rank models by a specific metric."""
    ranked = df.sort_values(metric, ascending=ascending).copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def find_top_models(
    df: pd.DataFrame, metrics: List[str], top_n: int = 10
) -> Dict[str, pd.DataFrame]:
    """Find top N models for each metric."""
    results = {}

    for metric in metrics:
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in dataframe")
            continue

        ascending = (
            "max_drawdown" in metric
            or metric.startswith("dh_min")
            or metric.startswith("bs_min")
        )

        ranked = rank_models(df, metric, ascending=ascending)
        top = ranked.head(top_n)

        results[metric] = top

    return results


def compare_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate improvements over Black-Scholes baseline."""
    comparison = df.copy()

    comparison["beats_bs_sharpe"] = (
        comparison["dh_sharpe_ratio"] > comparison["bs_sharpe_ratio"]
    )
    comparison["beats_bs_mean"] = comparison["dh_mean"] > comparison["bs_mean"]
    comparison["beats_bs_cvar"] = comparison["dh_cvar_95"] > comparison["bs_cvar_95"]

    comparison["beats_bs_count"] = (
        comparison["beats_bs_sharpe"].astype(int)
        + comparison["beats_bs_mean"].astype(int)
        + comparison["beats_bs_cvar"].astype(int)
    )

    return comparison.sort_values("beats_bs_count", ascending=False)


def pareto_frontier(
    df: pd.DataFrame,
    objective1: str,
    objective2: str,
    maximize_obj1: bool = True,
    maximize_obj2: bool = True,
) -> pd.DataFrame:
    """Find Pareto-optimal models for two objectives."""
    pareto_models = []

    for idx, row in df.iterrows():
        is_dominated = False

        for _, other_row in df.iterrows():
            if idx == other_row.name:
                continue

            obj1_better = (
                (other_row[objective1] > row[objective1])
                if maximize_obj1
                else (other_row[objective1] < row[objective1])
            )
            obj2_better = (
                (other_row[objective2] > row[objective2])
                if maximize_obj2
                else (other_row[objective2] < row[objective2])
            )

            if obj1_better and obj2_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_models.append(idx)

    return df.loc[pareto_models].sort_values(objective1, ascending=not maximize_obj1)


def filter_by_criteria(df: pd.DataFrame, **criteria) -> pd.DataFrame:
    """Filter models meeting specific criteria."""
    filtered = df.copy()

    for key, value in criteria.items():
        if key not in filtered.columns:
            logger.warning(f"Criteria '{key}' not found in dataframe")
            continue

        if isinstance(value, (list, tuple)):
            filtered = filtered[filtered[key].isin(value)]
        else:
            filtered = filtered[filtered[key] == value]

    return filtered


def analyze_architecture_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze average performance by architecture."""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    grouped = df.groupby("model_type")[numeric_cols].mean()

    key_metrics = [
        col
        for col in [
            "dh_sharpe_ratio",
            "dh_mean",
            "dh_cvar_95",
            "dh_win_rate",
            "dh_sharpe_improvement",
        ]
        if col in grouped.columns
    ]

    return grouped[key_metrics].sort_values("dh_sharpe_ratio", ascending=False)


def analyze_hyperparameter_impact(
    df: pd.DataFrame, param: str, metric: str = "dh_sharpe_ratio"
) -> pd.DataFrame:
    """Analyze how a hyperparameter affects performance."""
    if param not in df.columns or metric not in df.columns:
        logger.warning(f"Parameter '{param}' or metric '{metric}' not found")
        return pd.DataFrame()

    grouped = df.groupby(param)[metric].agg(["mean", "std", "min", "max", "count"])
    grouped = grouped.sort_values("mean", ascending=False)

    return grouped


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare and rank hyperparameter tuning results"
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
        default="hparam_tuning/analysis/rankings",
        help="Output directory for rankings",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top models to show"
    )
    args = parser.parse_args()

    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results from {results_file}")
    df = pd.read_pickle(results_file)
    logger.info(f"Loaded {len(df)} models")

    key_metrics = [
        "dh_sharpe_ratio",
        "dh_mean",
        "dh_cvar_95",
        "dh_sortino_ratio",
        "dh_win_rate",
        "dh_sharpe_improvement",
        "dh_mean_improvement",
    ]

    logger.info(f"\n{'='*60}")
    logger.info("Finding Top Models by Different Metrics")
    logger.info(f"{'='*60}")

    top_models = find_top_models(df, key_metrics, top_n=args.top_n)

    for metric, top_df in top_models.items():
        output_file = output_dir / f"top_{args.top_n}_by_{metric}.csv"
        top_df.to_csv(output_file, index=False)
        logger.info(f"\nTop {args.top_n} by {metric}:")
        display_cols = [
            "model_dir",
            "model_type",
            "n_layers",
            "n_units",
            "risk_measure",
            metric,
        ]
        display_cols = [c for c in display_cols if c in top_df.columns]
        logger.info(f"\n{top_df[display_cols].to_string(index=False)}")
        logger.info(f"  Saved to {output_file}")

    logger.info(f"\n{'='*60}")
    logger.info("Comparison vs Black-Scholes Baseline")
    logger.info(f"{'='*60}")

    comparison = compare_vs_baseline(df)
    comparison_file = output_dir / "best_vs_baseline.csv"
    comparison.to_csv(comparison_file, index=False)

    beats_all = comparison[comparison["beats_bs_count"] == 3]
    logger.info(
        f"\nModels beating BS on all 3 metrics (Sharpe, Mean PnL, CVaR): {len(beats_all)}"
    )
    if len(beats_all) > 0:
        logger.info(
            f"\n{beats_all[['model_dir', 'model_type', 'dh_sharpe_improvement', 'dh_mean_improvement', 'dh_cvar_improvement']].head(10).to_string(index=False)}"
        )

    beats_sharpe = comparison[comparison["beats_bs_sharpe"]].shape[0]
    beats_mean = comparison[comparison["beats_bs_mean"]].shape[0]
    beats_cvar = comparison[comparison["beats_bs_cvar"]].shape[0]

    logger.info(f"\nSummary:")
    logger.info(
        f"  Models beating BS Sharpe: {beats_sharpe}/{len(df)} ({100*beats_sharpe/len(df):.1f}%)"
    )
    logger.info(
        f"  Models beating BS Mean PnL: {beats_mean}/{len(df)} ({100*beats_mean/len(df):.1f}%)"
    )
    logger.info(
        f"  Models beating BS CVaR: {beats_cvar}/{len(df)} ({100*beats_cvar/len(df):.1f}%)"
    )
    logger.info(f"  Saved to {comparison_file}")

    logger.info(f"\n{'='*60}")
    logger.info("Pareto Frontier Analysis")
    logger.info(f"{'='*60}")

    pareto_sharpe_cvar = pareto_frontier(
        df, "dh_sharpe_ratio", "dh_cvar_95", maximize_obj1=True, maximize_obj2=True
    )
    pareto_file = output_dir / "pareto_sharpe_vs_cvar.csv"
    pareto_sharpe_cvar.to_csv(pareto_file, index=False)

    logger.info(f"\nPareto frontier (Sharpe vs CVaR): {len(pareto_sharpe_cvar)} models")
    logger.info(
        f"\n{pareto_sharpe_cvar[['model_dir', 'model_type', 'n_layers', 'n_units', 'dh_sharpe_ratio', 'dh_cvar_95']].head(10).to_string(index=False)}"
    )
    logger.info(f"  Saved to {pareto_file}")

    logger.info(f"\n{'='*60}")
    logger.info("Architecture Performance Analysis")
    logger.info(f"{'='*60}")

    arch_perf = analyze_architecture_performance(df)
    arch_file = output_dir / "performance_by_architecture.csv"
    arch_perf.to_csv(arch_file)

    logger.info(f"\nAverage performance by model type:")
    logger.info(f"\n{arch_perf.to_string()}")
    logger.info(f"  Saved to {arch_file}")

    logger.info(f"\n{'='*60}")
    logger.info("Hyperparameter Impact Analysis")
    logger.info(f"{'='*60}")

    for param in ["n_layers", "n_units", "risk_measure", "risk_param"]:
        impact = analyze_hyperparameter_impact(df, param, "dh_sharpe_ratio")
        if not impact.empty:
            impact_file = output_dir / f"impact_of_{param}.csv"
            impact.to_csv(impact_file)
            logger.info(f"\nImpact of {param} on Sharpe Ratio:")
            logger.info(f"\n{impact.to_string()}")
            logger.info(f"  Saved to {impact_file}")

    logger.info(f"\n{'='*60}")
    logger.info("Overall Best Model Recommendation")
    logger.info(f"{'='*60}")

    best_sharpe = df.nlargest(1, "dh_sharpe_ratio").iloc[0]
    best_improvement = df.nlargest(1, "dh_sharpe_improvement").iloc[0]

    logger.info(f"\nBest model by absolute Sharpe Ratio:")
    logger.info(f"  Model: {best_sharpe['model_dir']}")
    logger.info(
        f"  Type: {best_sharpe['model_type']}, Layers: {best_sharpe['n_layers']}, Units: {best_sharpe['n_units']}"
    )
    logger.info(f"  Risk: {best_sharpe['risk_measure']} ({best_sharpe['risk_param']})")
    logger.info(f"  Sharpe: {best_sharpe['dh_sharpe_ratio']:.4f}")
    logger.info(f"  Mean PnL: {best_sharpe['dh_mean']:.2f}")
    logger.info(f"  CVaR 95%: {best_sharpe['dh_cvar_95']:.2f}")

    logger.info(f"\nBest model by improvement over baseline:")
    logger.info(f"  Model: {best_improvement['model_dir']}")
    logger.info(
        f"  Type: {best_improvement['model_type']}, Layers: {best_improvement['n_layers']}, Units: {best_improvement['n_units']}"
    )
    logger.info(
        f"  Risk: {best_improvement['risk_measure']} ({best_improvement['risk_param']})"
    )
    logger.info(
        f"  Sharpe improvement: {best_improvement['dh_sharpe_improvement']:.4f}"
    )
    logger.info(
        f"  DH Sharpe: {best_improvement['dh_sharpe_ratio']:.4f}, BS Sharpe: {best_improvement['bs_sharpe_ratio']:.4f}"
    )

    logger.info(f"\nAll ranking files saved to {output_dir}/")


if __name__ == "__main__":
    main()
