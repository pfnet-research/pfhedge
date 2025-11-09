#!/usr/bin/env python3
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_model_name(dir_name: str) -> Optional[Dict[str, any]]:
    """Parse hyperparameters from model directory name."""
    pattern = r"^([a-f0-9]+)_(mlp|lstm|gru)_l(\d+)_u(\d+)_(entropic|expected_shortfall)([\d.]+)_lr([\d.]+)$"
    match = re.match(pattern, dir_name)

    if not match:
        return None

    run_id, model_type, layers, units, risk_measure, risk_param, lr = match.groups()

    return {
        "run_id": run_id,
        "model_type": model_type,
        "n_layers": int(layers),
        "n_units": int(units),
        "risk_measure": risk_measure,
        "risk_param": float(risk_param),
        "learning_rate": float(lr),
    }


def load_training_results(train_dir: Path) -> Optional[Dict[str, any]]:
    """Load training results from training_results.json."""
    results_file = train_dir / "training_results.json"

    if not results_file.exists():
        return None

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        test_metrics = summary.get("test_metrics", {})

        return {
            "train_n_epochs": summary.get("n_epochs"),
            "train_initial_loss": summary.get("initial_loss"),
            "train_final_loss": summary.get("final_loss"),
            "train_improvement_pct": summary.get("improvement_pct"),
            "train_created_at": summary.get("created_at"),
            "test_pnl_mean": test_metrics.get("pnl_mean"),
            "test_pnl_std": test_metrics.get("pnl_std"),
            "test_pnl_min": test_metrics.get("pnl_min"),
            "test_pnl_max": test_metrics.get("pnl_max"),
            "test_sharpe": test_metrics.get("sharpe"),
        }
    except Exception as e:
        logger.warning(f"Error loading {results_file}: {e}")
        return None


def load_backtest_results(backtest_dir: Path) -> Optional[Dict[str, any]]:
    """Load backtest results from results.json."""
    results_file = backtest_dir / "results.json"

    if not results_file.exists():
        return None

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        summary = data.get("summary", {})
        deep_hedge = summary.get("deep_hedge", {})
        bs_baseline = summary.get("bs_baseline", {})

        result = {
            "backtest_n_paths": data.get("n_paths"),
            "backtest_n_steps": data.get("n_steps"),
        }

        for metric in [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "sharpe_ratio",
            "sortino_ratio",
            "cvar_95",
            "var_95",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
        ]:
            result[f"dh_{metric}"] = deep_hedge.get(metric)
            result[f"bs_{metric}"] = bs_baseline.get(metric)

        result["dh_sharpe_improvement"] = deep_hedge.get(
            "sharpe_ratio", 0
        ) - bs_baseline.get("sharpe_ratio", 0)
        result["dh_mean_improvement"] = deep_hedge.get("mean", 0) - bs_baseline.get(
            "mean", 0
        )
        result["dh_cvar_improvement"] = deep_hedge.get("cvar_95", 0) - bs_baseline.get(
            "cvar_95", 0
        )

        # Calculate variance ratio (key metric for identifying actual hedgers)
        dh_std = deep_hedge.get("std", 1)
        bs_std = bs_baseline.get("std", 1)
        result["variance_ratio"] = dh_std / bs_std if bs_std > 0 else 999

        return result
    except Exception as e:
        logger.warning(f"Error loading {results_file}: {e}")
        return None


def find_subdirs(model_dir: Path, prefix: str) -> Optional[Path]:
    """Find subdirectory with given prefix (e.g., 'train_', 'backtest_')."""
    subdirs = [d for d in model_dir.glob(f"{prefix}*") if d.is_dir()]
    if subdirs:
        return subdirs[0]
    return None


def collect_all_results(hparam_dir: Path) -> pd.DataFrame:
    """Collect results from all model directories."""
    results = []

    model_dirs = [d for d in hparam_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(model_dirs)} model directories")

    for model_dir in sorted(model_dirs):
        hparams = parse_model_name(model_dir.name)

        if hparams is None:
            logger.warning(f"Skipping {model_dir.name} - cannot parse")
            continue

        train_dir = find_subdirs(model_dir, "train_")
        backtest_dir = find_subdirs(model_dir, "backtest_")

        if train_dir is None or backtest_dir is None:
            logger.warning(f"Skipping {model_dir.name} - missing subdirectories")
            continue

        train_results = load_training_results(train_dir)
        backtest_results = load_backtest_results(backtest_dir)

        if train_results is None or backtest_results is None:
            logger.warning(f"Skipping {model_dir.name} - missing results")
            continue

        row = {
            "model_dir": model_dir.name,
            **hparams,
            **train_results,
            **backtest_results,
        }

        results.append(row)
        logger.info(f"Loaded {model_dir.name}")

    df = pd.DataFrame(results)
    logger.info(f"\nSuccessfully loaded {len(df)} models")

    # Filter to only actual hedgers (variance ratio <= 2)
    total_models = len(df)
    actual_hedgers = df[df["variance_ratio"] <= 2.0]
    non_hedgers = df[df["variance_ratio"] > 2.0]

    logger.info(f"\n{'='*60}")
    logger.info(f"HEDGING ACTIVITY FILTER")
    logger.info(f"{'='*60}")
    logger.info(f"  Total models: {total_models}")
    logger.info(
        f"  Actual hedgers (variance ratio ≤ 2.0): {len(actual_hedgers)} ({100*len(actual_hedgers)/total_models:.1f}%)"
    )
    logger.info(
        f"  Non-hedgers (variance ratio > 2.0): {len(non_hedgers)} ({100*len(non_hedgers)/total_models:.1f}%)"
    )
    logger.info(f"\n  ⚠️  Filtering to ACTUAL HEDGERS ONLY for analysis")
    logger.info(
        f"      Non-hedging models hold near-zero positions and don't reduce risk"
    )

    if len(actual_hedgers) == 0:
        logger.warning(
            f"\n⚠️  WARNING: No models are actually hedging! All models increased variance."
        )
        logger.warning(
            f"   Proceeding with all models, but results may not represent true hedging."
        )
        return df

    return actual_hedgers


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate hyperparameter tuning results"
    )
    parser.add_argument(
        "--hparam-dir",
        type=str,
        default="hparam_tuning",
        help="Directory containing model results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hparam_tuning/analysis",
        help="Output directory for aggregated results",
    )
    args = parser.parse_args()

    hparam_dir = Path(args.hparam_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Aggregating results from {hparam_dir}")
    df = collect_all_results(hparam_dir)

    csv_path = output_dir / "all_results.csv"
    pkl_path = output_dir / "all_results.pkl"

    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    logger.info(f"\nSaved results to:")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  Pickle: {pkl_path}")

    logger.info(f"\nDataFrame shape (filtered): {df.shape}")

    logger.info("\nBasic statistics (actual hedgers only):")
    logger.info(f"  Model types: {df['model_type'].value_counts().to_dict()}")
    logger.info(f"  Layers: {sorted(df['n_layers'].unique())}")
    logger.info(f"  Units: {sorted(df['n_units'].unique())}")
    logger.info(f"  Risk measures: {df['risk_measure'].value_counts().to_dict()}")

    logger.info("\nTop 5 ACTUAL HEDGERS by Sharpe Ratio:")
    top5 = df.nlargest(5, "dh_sharpe_ratio")[
        [
            "model_dir",
            "model_type",
            "n_layers",
            "n_units",
            "risk_measure",
            "dh_sharpe_ratio",
            "variance_ratio",
        ]
    ]
    logger.info(f"\n{top5.to_string(index=False)}")


if __name__ == "__main__":
    main()
