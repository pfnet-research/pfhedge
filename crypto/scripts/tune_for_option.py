#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Optional YAML support for reading templates/search spaces
try:
    import yaml  # type: ignore

    HAS_YAML = True
except Exception:
    HAS_YAML = False


# ------------------------------
# Utility helpers
# ------------------------------


def _now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now().isoformat(timespec="seconds")


def _sha1_short(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def _slugify(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace(" ", "_")
        .replace("=", "-")
        .replace(",", "-")
        .replace("..", ".")
    )


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_executable(path: Path) -> None:
    try:
        os.chmod(path, os.stat(path).st_mode | 0o111)
    except Exception:
        pass


def _find_actual_output_dir(base_output_dir: Path) -> Path:
    if base_output_dir.exists():
        return base_output_dir

    parent = base_output_dir.parent
    prefix = base_output_dir.name

    if not parent.exists():
        return base_output_dir

    candidates = [d for d in parent.glob(prefix + "_*") if d.is_dir()]
    if not candidates:
        return base_output_dir

    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_yaml_if_available(path: Path) -> Optional[Dict[str, Any]]:
    if not HAS_YAML:
        return None
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


# ------------------------------
# Default search space
# ------------------------------

DEFAULT_SEARCH_SPACE: Dict[str, Any] = {
    # Model family and size
    "model_type": ["gru", "lstm"],
    "layers": [2, 3],
    "units": [32, 64, 128],
    # Risk objective and parameters (measure-specific)
    "risk_measure": ["entropic", "expected_shortfall"],
    "risk_param_by_measure": {
        "entropic": [1.0, 2.0, 3.0],
        "expected_shortfall": [0.85, 0.90, 0.95],
    },
    # Optimizer configs per risk measure (optional, falls back to global if not specified)
    "optimizer_by_measure": {
        "entropic": [
            {"optimizer": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-3}
        ],
        "expected_shortfall": [
            {"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-4}
        ],
    },
    # Volatility signal
    "volatility_window": [0, 20, 40],
    # Training budget
    "epochs": [100],
    "paths": [500000],
}


# ------------------------------
# CLI parsing
# ------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning for train_for_option.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Option selection
    p.add_argument(
        "--option-file", required=True, help="JSON file from explore_options.py"
    )
    p.add_argument(
        "--instrument",
        required=True,
        help="Instrument name (e.g., BTC-31OCT25-110000-C)",
    )

    # Output and execution
    p.add_argument(
        "--output-root", default="results/hparam_tuning", help="Root directory for runs"
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "auto"],
        help="Device for training",
    )
    p.add_argument(
        "--resume", action="store_true", help="Resume from previous tuning_state.json"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts/configs but do not execute",
    )
    p.add_argument(
        "--run-backtest",
        action="store_true",
        help="Run backtest after training each run",
    )

    # Search space
    p.add_argument("--search-space", help="YAML/JSON file specifying search space")
    p.add_argument("--max-runs", type=int, help="Limit number of runs from the grid")

    # Training invariants (these apply across the grid unless overridden by the grid itself)
    p.add_argument(
        "--underlying",
        default="spot",
        choices=["spot", "perpetual"],
        help="Underlying type",
    )
    p.add_argument(
        "--cost",
        type=float,
        default=None,
        help="Transaction cost for training. If not set, uses default per underlying (spot=0.001, perpetual=0.0006)",
    )
    p.add_argument("--dt-hours", type=float, default=8.0, help="Time step in hours")
    p.add_argument("--seed", type=int, default=42, help="Random seed for training")

    # Backtest template / overrides
    p.add_argument(
        "--backtest-template", help="Path to YAML template (e.g., backtest.yaml)"
    )
    p.add_argument("--bt-start-date", help="Override backtest start_date")
    p.add_argument("--bt-end-date", help="Override backtest end_date")
    p.add_argument("--bt-data-dir", help="Override backtest data_dir")
    p.add_argument("--bt-data-file", help="Override backtest data_file")
    p.add_argument(
        "--bt-n-paths", type=int, default=100, help="Backtest bootstrap paths"
    )

    return p.parse_args()


# ------------------------------
# State management
# ------------------------------


class TuningState:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "tuning_state.json"
        self.state: Dict[str, Any] = {"runs": {}, "created_at": _now_iso()}
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text())
            except Exception:
                pass

    def save(self) -> None:
        _write_text(self.path, json.dumps(self.state, indent=2))

    def get(self, run_id: str) -> Dict[str, Any]:
        return self.state["runs"].get(run_id, {})

    def update_run(self, run_id: str, info: Dict[str, Any]) -> None:
        self.state.setdefault("runs", {})
        run = self.state["runs"].get(run_id, {})
        run.update(info)
        self.state["runs"][run_id] = run
        self.save()


# ------------------------------
# Grid generation
# ------------------------------


def load_search_space(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_SEARCH_SPACE

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Search space file not found: {path}")

    # Try YAML first, then JSON
    if HAS_YAML:
        try:
            with open(p, "r") as f:
                return yaml.safe_load(f)  # type: ignore
        except Exception:
            pass

    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise ValueError(f"Failed to parse search space file {path}: {e}")


def expand_grid(space: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Check if using measure-specific optimizer configs
    optimizer_by_measure = space.get("optimizer_by_measure", {})
    use_measure_specific_optimizers = bool(optimizer_by_measure)

    if use_measure_specific_optimizers:
        # Per-measure optimizer configs
        axes = {
            "model_type": space.get("model_type", ["gru"]),
            "layers": space.get("layers", [2]),
            "units": space.get("units", [32]),
            "risk_measure": space.get("risk_measure", ["entropic"]),
            "volatility_window": space.get("volatility_window", [20]),
            "epochs": space.get("epochs", [100]),
            "paths": space.get("paths", [500000]),
        }
    else:
        # Global optimizer configs
        axes = {
            "model_type": space.get("model_type", ["gru"]),
            "layers": space.get("layers", [2]),
            "units": space.get("units", [32]),
            "risk_measure": space.get("risk_measure", ["entropic"]),
            "optimizer": space.get("optimizer", ["adamw"]),
            "learning_rate": space.get("learning_rate", [1e-3]),
            "weight_decay": space.get("weight_decay", [1e-4]),
            "volatility_window": space.get("volatility_window", [20]),
            "epochs": space.get("epochs", [100]),
            "paths": space.get("paths", [500000]),
        }

    risk_param_by_measure = space.get("risk_param_by_measure", {"entropic": [2.0]})

    combos: List[Dict[str, Any]] = []

    if use_measure_specific_optimizers:
        # Expand grid with measure-specific optimizers
        for model_type, layers, units, risk_measure, vol_win, epochs, paths in product(
            axes["model_type"],
            axes["layers"],
            axes["units"],
            axes["risk_measure"],
            axes["volatility_window"],
            axes["epochs"],
            axes["paths"],
        ):
            rparams = risk_param_by_measure.get(risk_measure, [1.0])
            optimizer_configs = optimizer_by_measure.get(
                risk_measure,
                [{"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-4}],
            )

            for rparam in rparams:
                for opt_cfg in optimizer_configs:
                    combos.append(
                        {
                            "model_type": model_type,
                            "layers": int(layers),
                            "units": int(units),
                            "risk_measure": risk_measure,
                            "risk_param": float(rparam),
                            "optimizer": opt_cfg["optimizer"],
                            "learning_rate": float(opt_cfg["learning_rate"]),
                            "weight_decay": float(opt_cfg["weight_decay"]),
                            "volatility_window": int(vol_win),
                            "epochs": int(epochs),
                            "paths": int(paths),
                        }
                    )
    else:
        # Original global optimizer grid expansion
        for (
            model_type,
            layers,
            units,
            risk_measure,
            optimizer,
            lr,
            wd,
            vol_win,
            epochs,
            paths,
        ) in product(
            axes["model_type"],
            axes["layers"],
            axes["units"],
            axes["risk_measure"],
            axes["optimizer"],
            axes["learning_rate"],
            axes["weight_decay"],
            axes["volatility_window"],
            axes["epochs"],
            axes["paths"],
        ):
            rparams = risk_param_by_measure.get(risk_measure, [1.0])
            for rparam in rparams:
                combos.append(
                    {
                        "model_type": model_type,
                        "layers": int(layers),
                        "units": int(units),
                        "risk_measure": risk_measure,
                        "risk_param": float(rparam),
                        "optimizer": optimizer,
                        "learning_rate": float(lr),
                        "weight_decay": float(wd),
                        "volatility_window": int(vol_win),
                        "epochs": int(epochs),
                        "paths": int(paths),
                    }
                )

    # Stable sort for reproducibility
    combos.sort(key=lambda d: json.dumps(d, sort_keys=True))
    return combos


# ------------------------------
# Training execution
# ------------------------------


def build_train_command(
    option_file: str,
    instrument: str,
    device: str,
    underlying: str,
    transaction_cost: float,
    dt_hours: float,
    seed: int,
    output_dir: Path,
    hp: Dict[str, Any],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "crypto" / "scripts" / "train_for_option.py"),
        "--option-file",
        option_file,
        "--instrument",
        instrument,
        "--model-type",
        str(hp["model_type"]),
        "--layers",
        str(hp["layers"]),
        "--units",
        str(hp["units"]),
        "--risk-measure",
        str(hp["risk_measure"]),
        "--risk-param",
        str(hp["risk_param"]),
        "--underlying",
        underlying,
        "--cost",
        str(transaction_cost),
        "--dt-hours",
        str(dt_hours),
        "--volatility-window",
        str(hp["volatility_window"]),
        "--epochs",
        str(hp["epochs"]),
        "--paths",
        str(hp["paths"]),
        "--seed",
        str(seed),
        "--device",
        device,
        "--optimizer",
        str(hp["optimizer"]),
        "--learning-rate",
        str(hp["learning_rate"]),
        "--weight-decay",
        str(hp["weight_decay"]),
        "--output",
        str(output_dir),
    ]
    return cmd


def run_subprocess(cmd: List[str], log_path: Path) -> int:
    with open(log_path, "w") as logf:
        logf.write(f"# Command: {' '.join(shlex.quote(c) for c in cmd)}\n")
        logf.write(f"# Started: {_now_iso()}\n\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        returncode = proc.wait()
        logf.write(f"\n# Finished: {_now_iso()} (code={returncode})\n")
    return returncode


# ------------------------------
# Backtest config generation
# ------------------------------


def _read_backtest_template(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not HAS_YAML:
        print("Warning: PyYAML not installed; ignoring backtest template.")
        return {}
    tpl = _load_yaml_if_available(Path(path))
    return tpl or {}


def build_backtest_config(
    template: Dict[str, Any],
    *,
    model_path: str,
    option_metadata: Dict[str, Any],
    underlying: str,
    transaction_cost: float,
    dt_hours: float,
    n_bootstrap_paths: int,
    output_dir: Path,
    overrides: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(template) if template else {}

    # Required
    if overrides.get("start_date"):
        cfg["start_date"] = overrides["start_date"]
    if overrides.get("end_date"):
        cfg["end_date"] = overrides["end_date"]

    # If template lacked dates and overrides are empty, keep as-is; user can edit later.

    # Option fields from metadata (absolute strike, maturity, call)
    cfg["strike"] = option_metadata["strike"]
    cfg["maturity_days"] = option_metadata["days_to_expiry"]
    cfg["call"] = option_metadata["option_type"] == "call"

    # Execution
    cfg["model_path"] = model_path
    cfg["n_bootstrap_paths"] = n_bootstrap_paths
    cfg["transaction_cost"] = transaction_cost
    cfg["dt_hours"] = dt_hours
    cfg["underlying_type"] = underlying

    # Data paths
    if overrides.get("data_dir"):
        cfg["data_dir"] = overrides["data_dir"]
    if overrides.get("data_file"):
        cfg["data_file"] = overrides["data_file"]

    # Output dir per run
    cfg["output_dir"] = str(output_dir)

    # Bootstrap mode to preserve moneyness
    cfg["bootstrap_mode"] = "normalize_spot"
    initial_spot = option_metadata.get("initial_spot")
    if initial_spot is not None:
        cfg["initial_spot"] = float(initial_spot)

    # Propagate volatility_window if present in template (optional)
    if "volatility_window" not in cfg:
        cfg["volatility_window"] = 20

    return cfg


def save_backtest_yaml(cfg: Dict[str, Any], path: Path) -> None:
    if not HAS_YAML:
        # Fallback: minimal JSON-like dump; user can run CLI without YAML too
        _write_text(path, json.dumps(cfg, indent=2))
        return
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)  # type: ignore


# ------------------------------
# Main orchestration
# ------------------------------


def main() -> int:
    args = parse_args()

    output_root = Path(args.output_root).resolve()
    _mkdir(output_root)

    # Determine training transaction cost by underlying if not specified
    if args.cost is None:
        transaction_cost = 0.001 if args.underlying == "spot" else 0.0006
    else:
        transaction_cost = args.cost

    # Load/prepare grid
    space = load_search_space(args.search_space)
    grid = expand_grid(space)

    # Print grid size and warn if large
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    print(f"Grid contains {len(grid)} configuration(s)")
    print(f"Option: {args.instrument}")
    print(f"Device: {args.device}")
    print(f"Output: {output_root}")

    if args.max_runs is not None:
        grid = grid[: args.max_runs]
        print(f"Limited to first {len(grid)} runs (--max-runs={args.max_runs})")

    if len(grid) > 50 and not args.resume and not args.dry_run:
        print(f"\n⚠️  WARNING: Large grid with {len(grid)} runs detected!")
        print(f"   This may take a long time. Consider:")
        print(f"   - Use --max-runs to limit the grid")
        print(f"   - Use --dry-run to generate scripts without running")
        print(f"   - Use --resume to skip completed runs")
        try:
            response = input("\nPress Enter to continue or Ctrl+C to abort: ")
        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            return 1

    print(f"{'='*60}\n")

    # Load backtest template and overrides
    bt_template = _read_backtest_template(args.backtest_template)
    bt_overrides = {
        "start_date": args.bt_start_date,
        "end_date": args.bt_end_date,
        "data_dir": args.bt_data_dir,
        "data_file": args.bt_data_file,
    }

    # State management
    state = TuningState(output_root)

    # Summary CSV header
    summary_csv = output_root / "summary.csv"
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(
                [
                    "run_id",
                    "status",
                    "model_type",
                    "layers",
                    "units",
                    "risk_measure",
                    "risk_param",
                    "optimizer",
                    "learning_rate",
                    "weight_decay",
                    "volatility_window",
                    "epochs",
                    "paths",
                    "model_path",
                    "train_dir",
                    "backtest_config",
                    "backtest_output_dir",
                ]
            )

    # Iterate runs sequentially
    for hp in grid:
        run_key_payload = {
            "instrument": args.instrument,
            "option_file": args.option_file,
            "underlying": args.underlying,
            "seed": args.seed,
            **hp,
        }
        run_id = _sha1_short(run_key_payload)
        human_tag = _slugify(
            f"{hp['model_type']}_l{hp['layers']}_u{hp['units']}_{hp['risk_measure']}{hp['risk_param']}_lr{hp['learning_rate']}"
        )

        run_dir = output_root / f"{run_id}_{human_tag}"
        train_base_dir = run_dir / "train"
        backtest_dir = run_dir / "backtest"
        _mkdir(run_dir)
        _mkdir(train_base_dir)
        _mkdir(backtest_dir)

        # If resume, check if model already exists
        train_actual_dir = _find_actual_output_dir(train_base_dir)
        model_path = train_actual_dir / "model.pth"
        has_trained = model_path.exists()
        has_backtested = (backtest_dir / "results.json").exists()

        if args.resume:
            if has_trained and has_backtested:
                state.update_run(
                    run_id,
                    {
                        "status": "completed",
                        "model_path": str(model_path),
                        "train_dir": str(train_actual_dir),
                        "backtest_output": str(backtest_dir),
                        "updated_at": _now_iso(),
                    },
                )
                continue

        # Write train command script
        train_cmd_list = build_train_command(
            option_file=args.option_file,
            instrument=args.instrument,
            device=args.device,
            underlying=args.underlying,
            transaction_cost=transaction_cost,
            dt_hours=args.dt_hours,
            seed=args.seed,
            output_dir=train_base_dir,
            hp=hp,
        )
        train_cmd_str = " ".join(shlex.quote(c) for c in train_cmd_list)
        train_cmd_sh = run_dir / "train_cmd.sh"
        _write_text(
            train_cmd_sh,
            "#!/usr/bin/env bash\nset -euo pipefail\n" + train_cmd_str + "\n",
        )
        _make_executable(train_cmd_sh)

        # Run training if needed
        if not has_trained and not args.dry_run:
            state.update_run(
                run_id,
                {
                    "status": "training",
                    "hyperparams": hp,
                    "created_at": _now_iso(),
                    "train_cmd": train_cmd_str,
                },
            )
            code = run_subprocess(train_cmd_list, log_path=run_dir / "train.log")
            if code != 0:
                state.update_run(
                    run_id,
                    {
                        "status": "failed_training",
                        "returncode": code,
                        "updated_at": _now_iso(),
                    },
                )
                continue

            # Refresh actual dir and model path (in case of git hash suffix)
            train_actual_dir = _find_actual_output_dir(train_base_dir)
            model_path = train_actual_dir / "model.pth"
            has_trained = model_path.exists()

        if not has_trained:
            # Skip backtest if model missing
            state.update_run(
                run_id, {"status": "missing_model", "updated_at": _now_iso()}
            )
            continue

        # Load option metadata saved by train_for_option.py
        option_meta_path = train_actual_dir / "option_metadata.json"
        option_metadata = _safe_load_json(option_meta_path) or {}

        # Build backtest config
        bt_cfg = build_backtest_config(
            bt_template,
            model_path=str(model_path),
            option_metadata=option_metadata,
            underlying=args.underlying,
            transaction_cost=transaction_cost,
            dt_hours=args.dt_hours,
            n_bootstrap_paths=int(args.bt_n_paths),
            output_dir=backtest_dir,
            overrides={
                "start_date": args.bt_start_date,
                "end_date": args.bt_end_date,
                "data_dir": args.bt_data_dir,
                "data_file": args.bt_data_file,
            },
        )

        bt_yaml_path = run_dir / "backtest.yaml"
        save_backtest_yaml(bt_cfg, bt_yaml_path)

        # Write backtest command script
        bt_cmd_list = [
            sys.executable,
            "-m",
            "crypto.backtest.run",
            "--config",
            str(bt_yaml_path),
            "--seed",
            str(args.seed),
        ]
        bt_cmd_str = " ".join(shlex.quote(c) for c in bt_cmd_list)
        bt_cmd_sh = run_dir / "backtest_cmd.sh"
        _write_text(
            bt_cmd_sh, "#!/usr/bin/env bash\nset -euo pipefail\n" + bt_cmd_str + "\n"
        )
        _make_executable(bt_cmd_sh)

        # Maybe run backtest
        if args.run_backtest and not has_backtested and not args.dry_run:
            state.update_run(
                run_id,
                {
                    "status": "backtesting",
                    "model_path": str(model_path),
                    "train_dir": str(train_actual_dir),
                    "backtest_cmd": bt_cmd_str,
                    "updated_at": _now_iso(),
                },
            )
            code = run_subprocess(bt_cmd_list, log_path=run_dir / "backtest.log")
            if code != 0:
                state.update_run(
                    run_id,
                    {
                        "status": "failed_backtest",
                        "returncode": code,
                        "updated_at": _now_iso(),
                    },
                )
                continue
            has_backtested = (backtest_dir / "results.json").exists()

        # Finalize status
        final_status = (
            "completed"
            if (has_trained and (has_backtested or not args.run_backtest))
            else "trained_only"
        )
        state.update_run(
            run_id,
            {
                "status": final_status,
                "model_path": str(model_path),
                "train_dir": str(train_actual_dir),
                "backtest_config": str(bt_yaml_path),
                "backtest_output": str(backtest_dir),
                "updated_at": _now_iso(),
            },
        )

        # Append to summary CSV
        with open(summary_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    run_id,
                    final_status,
                    hp["model_type"],
                    hp["layers"],
                    hp["units"],
                    hp["risk_measure"],
                    hp["risk_param"],
                    hp["optimizer"],
                    hp["learning_rate"],
                    hp["weight_decay"],
                    hp["volatility_window"],
                    hp["epochs"],
                    hp["paths"],
                    str(model_path),
                    str(train_actual_dir),
                    str(bt_yaml_path),
                    str(backtest_dir),
                ]
            )

    print(f"\n✓ Tuning complete. State: {state.path}\n   Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
