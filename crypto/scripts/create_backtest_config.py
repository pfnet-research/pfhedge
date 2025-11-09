#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
import yaml


def create_backtest_config(
    train_results_path: str,
    option_metadata_path: str,
    model_path: str,
    template_path: str,
    output_path: str,
    output_dir: str,
):
    with open(train_results_path) as f:
        train_results = json.load(f)

    option_metadata = {}
    if Path(option_metadata_path).exists():
        with open(option_metadata_path) as f:
            option_metadata = json.load(f)

    with open(template_path) as f:
        template = yaml.safe_load(f)

    config = train_results.get("model_config", {})

    repo_root = Path(__file__).parent.parent.parent.resolve()
    model_path_abs = Path(model_path).resolve()
    output_dir_abs = Path(output_dir).resolve()

    data_dir = template.get("data_dir", "crypto/data/historical")
    data_dir_abs = (
        (repo_root / data_dir).resolve()
        if not Path(data_dir).is_absolute()
        else Path(data_dir).resolve()
    )

    backtest_config = {
        "start_date": template.get("start_date", "2025-01-01"),
        "end_date": template.get("end_date", "2025-09-25"),
        "strike": option_metadata.get("strike", config.get("strike")),
        "maturity_days": config.get("maturity_days", 30),
        "call": config.get("call", True),
        "bootstrap_mode": "normalize_spot",
        "initial_spot": option_metadata.get("initial_spot", 109325.75),
        "model_path": str(model_path_abs),
        "n_bootstrap_paths": template.get("n_bootstrap_paths", 1000),
        "transaction_cost": config.get("transaction_cost", 0.001),
        "dt_hours": config.get("dt_hours", 8.0),
        "underlying_type": config.get("underlying_type", "spot"),
        "data_dir": str(data_dir_abs),
        "data_file": template.get("data_file"),
        "output_dir": str(output_dir_abs),
        "seed": template.get("seed", 42),
        "save_raw_data": template.get("save_raw_data", False),
    }

    with open(output_path, "w") as f:
        yaml.dump(backtest_config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created: {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create backtest config from training results"
    )
    parser.add_argument(
        "--train-results", required=True, help="Path to training_results.json"
    )
    parser.add_argument(
        "--option-metadata", required=True, help="Path to option_metadata.json"
    )
    parser.add_argument("--model-path", required=True, help="Path to model.pth")
    parser.add_argument(
        "--template", required=True, help="Path to backtest template YAML"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for backtest config"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for backtest results"
    )

    args = parser.parse_args()

    sys.exit(
        create_backtest_config(
            args.train_results,
            args.option_metadata,
            args.model_path,
            args.template,
            args.output,
            args.output_dir,
        )
    )
