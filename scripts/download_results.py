#!/usr/bin/env python3
"""
Download backtest results and trained models from remote GPU server.

Usage:
    python download_results.py <iteration> <ssh_host> <ssh_port> [--download-model]

Examples:
    python download_results.py 21 185.65.93.114 47612
    python download_results.py 22 185.65.93.114 47612 --download-model

Note:
    Model download decision should be made by analysis agent, not hardcoded logic.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, check=True):
    """Run shell command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        print(f"Warning: Command failed: {cmd}")
        print(f"  Error: {result.stderr}")
    return result.stdout, result.stderr, result.returncode


def download_file(ssh_host, ssh_port, remote_path, local_path, description):
    """Download a single file via scp."""
    print(f"  Downloading {description}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = f'scp -P {ssh_port} "root@{ssh_host}:{remote_path}" "{local_path}"'
    stdout, stderr, code = run_cmd(cmd, check=False)

    if code == 0:
        print(f"    ✓ {local_path}")
        return True
    else:
        print(f"    ✗ {description} not found")
        return False


def download_directory(ssh_host, ssh_port, remote_dir, local_dir, description):
    """Download entire directory via scp."""
    print(f"  Downloading {description}...")
    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = f'scp -r -P {ssh_port} "root@{ssh_host}:{remote_dir}/*" "{local_dir}/"'
    stdout, stderr, code = run_cmd(cmd, check=False)

    if code == 0:
        print(f"    ✓ {local_dir}")
        return True
    else:
        print(f"    ✗ {description} not found")
        return False


def get_model_directory(ssh_host, ssh_port, iteration):
    """Find model directory on remote server."""
    cmd = f'ssh -p {ssh_port} "root@{ssh_host}" "ls -d /workspace/pfhedge/models/iteration_{iteration}_* 2>/dev/null | head -1"'
    stdout, stderr, code = run_cmd(cmd, check=False)

    if code == 0 and stdout.strip():
        return stdout.strip()
    return None




def display_metrics_summary(metrics_file):
    """Display metrics summary."""
    if not metrics_file.exists():
        return

    try:
        with open(metrics_file) as f:
            metrics = json.load(f)

        dh = metrics.get("deep_hedge", {})
        print("\n=== Metrics Summary ===")
        print(f"Sharpe Ratio: {dh.get('sharpe_ratio', 'N/A')}")
        print(f"CVaR (95%): ${dh.get('cvar_95', 'N/A')}")
        print(f"Mean PnL: ${dh.get('mean_pnl', 'N/A')}")
        print(f"Win Rate: {dh.get('win_rate', 'N/A'):.1%}" if dh.get('win_rate') else "Win Rate: N/A")
    except Exception as e:
        print(f"Could not parse metrics: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download backtest results and models")
    parser.add_argument("iteration", type=int, help="Iteration number")
    parser.add_argument("ssh_host", help="SSH host")
    parser.add_argument("ssh_port", type=int, help="SSH port")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download trained model (decision should be made by analysis agent)",
    )
    args = parser.parse_args()

    iteration = args.iteration
    ssh_host = args.ssh_host
    ssh_port = args.ssh_port

    print(f"\n=== Downloading Iteration {iteration} Results ===")
    print(f"Remote: root@{ssh_host}:{ssh_port}\n")

    # Setup local directories
    local_results_dir = Path(f"results/iteration_{iteration}")
    local_models_dir = Path("models")
    local_results_dir.mkdir(parents=True, exist_ok=True)
    local_models_dir.mkdir(parents=True, exist_ok=True)

    # Download backtest results
    print("[1/5] Downloading backtest report...")
    download_file(
        ssh_host,
        ssh_port,
        f"/workspace/pfhedge/backtest_results/iteration_{iteration}/backtest_report.md",
        local_results_dir / "backtest_report.md",
        "report",
    )

    print("\n[2/5] Downloading metrics...")
    metrics_file = local_results_dir / "backtest_metrics.json"
    metrics_downloaded = download_file(
        ssh_host,
        ssh_port,
        f"/workspace/pfhedge/backtest_results/iteration_{iteration}/backtest_metrics.json",
        metrics_file,
        "metrics",
    )

    print("\n[3/5] Downloading raw data...")
    download_file(
        ssh_host,
        ssh_port,
        f"/workspace/pfhedge/backtest_results/iteration_{iteration}/raw_data.pkl",
        local_results_dir / "raw_data.pkl",
        "raw data",
    )

    print("\n[4/5] Downloading plots...")
    download_directory(
        ssh_host,
        ssh_port,
        f"/workspace/pfhedge/backtest_results/iteration_{iteration}/plots",
        local_results_dir / "plots",
        "plots",
    )

    # Download model if requested
    print("\n[5/5] Checking model download request...")

    if args.download_model:
        print(f"\n  Downloading model...")

        model_dir_remote = get_model_directory(ssh_host, ssh_port, iteration)
        if not model_dir_remote:
            print(f"  Error: No model directory found for iteration {iteration}")
            sys.exit(1)

        model_name = Path(model_dir_remote).name
        local_model_dir = local_models_dir / model_name

        print(f"  Model: {model_name}")

        # Download model files
        download_file(
            ssh_host,
            ssh_port,
            f"{model_dir_remote}/model.pth",
            local_model_dir / "model.pth",
            "model weights",
        )

        download_file(
            ssh_host,
            ssh_port,
            f"{model_dir_remote}/config.yaml",
            local_model_dir / "config.yaml",
            "config",
        )

        download_file(
            ssh_host,
            ssh_port,
            f"{model_dir_remote}/training_log.txt",
            local_model_dir / "training_log.txt",
            "training log",
        )

        print(f"\n  Model saved to: {local_model_dir}")
    else:
        print("  Model download not requested")
        print("  (Use --download-model flag to download model)")

    # Display summary
    print("\n=== Download Complete ===")
    print(f"Results location: {local_results_dir}")

    display_metrics_summary(metrics_file)

    print()


if __name__ == "__main__":
    main()
