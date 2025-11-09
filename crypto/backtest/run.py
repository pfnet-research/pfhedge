#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
from typing import Optional

from crypto.backtest.config import BacktestConfig
from crypto.backtest.backtester import Backtester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep hedging backtest on historical Bitcoin data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML config file (optional if all params provided via CLI)",
    )

    # Required parameters (if no config file)
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--strike", type=float, help="Option strike price")
    parser.add_argument("--maturity-days", type=int, help="Option maturity in days")
    parser.add_argument(
        "--model-path", type=str, help="Path to model checkpoint (.pth)"
    )

    # Optional parameters
    parser.add_argument(
        "--call",
        action="store_true",
        default=None,
        help="Call option (default: True)",
    )
    parser.add_argument(
        "--put", action="store_true", help="Put option (overrides --call)"
    )
    parser.add_argument(
        "--n-bootstrap-paths",
        "--n-paths",
        type=int,
        help="Number of bootstrap paths (default: 100)",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        help="Transaction cost rate (default: 0.0005)",
    )
    parser.add_argument(
        "--dt-hours", type=float, help="Time step in hours (default: 8.0)"
    )
    parser.add_argument(
        "--data-dir", type=str, help="Data directory (default: sample_data)"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory (default: backtest_results)"
    )

    # Execution parameters
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility (recommended)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating report (only save results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and print resolved configuration without running backtest",
    )
    parser.add_argument(
        "--print-default-config",
        action="store_true",
        help="Print a minimal YAML configuration template and exit",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--no-insights",
        action="store_true",
        help="Skip printing key insights summary",
    )
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="Use plain text without emoji decorations",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable MLP input/output diagnostics during hedging",
    )
    parser.add_argument(
        "--save-raw-data",
        action="store_true",
        help="Save raw timeseries data (positions, PnL, spots) to JSON. Warning: can create large files",
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> BacktestConfig:
    # Load from YAML if provided
    if args.config:
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")

        print(f"Loading config from {args.config}")
        config = BacktestConfig.load_yaml(args.config)

        # Override with CLI parameters if provided
        if args.start_date:
            config.start_date = args.start_date
        if args.end_date:
            config.end_date = args.end_date
        if args.strike is not None:
            config.strike = args.strike
        if args.maturity_days is not None:
            config.maturity_days = args.maturity_days
        if args.model_path:
            config.model_path = args.model_path
        if args.put:
            config.call = False
        elif args.call is not None:
            config.call = args.call
        if args.n_bootstrap_paths is not None:
            config.n_bootstrap_paths = args.n_bootstrap_paths
        if args.transaction_cost is not None:
            config.transaction_cost = args.transaction_cost
        if args.dt_hours is not None:
            config.dt_hours = args.dt_hours
        if args.data_dir:
            config.data_dir = args.data_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.diagnostics is not None:
            config.enable_diagnostics = args.diagnostics
        if args.seed is not None:
            config.seed = args.seed
        # Only override if flag was explicitly provided (action="store_true" defaults to False, not None)
        if args.save_raw_data:
            config.save_raw_data = True

        return config

    # Create config from CLI parameters
    required_params = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "strike": args.strike,
        "maturity_days": args.maturity_days,
        "model_path": args.model_path,
    }

    missing = [name for name, value in required_params.items() if value is None]
    if missing:
        raise ValueError(
            f"Missing required parameters: {', '.join(missing)}. "
            f"Either provide --config or all required CLI parameters."
        )

    # Build config dict with defaults
    config_dict = required_params.copy()

    # Handle call/put
    if args.put:
        config_dict["call"] = False
    elif args.call is not None:
        config_dict["call"] = args.call
    # else: will use default (True)

    # Add optional parameters if provided
    if args.n_bootstrap_paths is not None:
        config_dict["n_bootstrap_paths"] = args.n_bootstrap_paths
    if args.transaction_cost is not None:
        config_dict["transaction_cost"] = args.transaction_cost
    if args.dt_hours is not None:
        config_dict["dt_hours"] = args.dt_hours
    if args.data_dir:
        config_dict["data_dir"] = args.data_dir
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.diagnostics is not None:
        config_dict["enable_diagnostics"] = args.diagnostics

    return BacktestConfig.from_dict(config_dict)


def main():
    try:
        args = parse_args()

        # Handle --print-default-config
        if args.print_default_config:
            print("# Minimal backtest configuration template")
            print("# Copy this to a file, edit the values, and run with --config")
            print()
            print("# Required parameters")
            print("start_date: '2024-01-01'      # Backtest start date (YYYY-MM-DD)")
            print("end_date: '2024-01-31'        # Backtest end date (YYYY-MM-DD)")
            print("strike: 50000                 # Option strike price in USD")
            print("maturity_days: 14             # Option maturity in days")
            print(
                "model_path: models/deep_hedger.pth  # Path to trained model checkpoint"
            )
            print()
            print("# Optional parameters (uncomment to override defaults)")
            print(
                "# call: true                    # true for call option, false for put"
            )
            print("# n_bootstrap_paths: 100        # Number of bootstrap paths")
            print("# transaction_cost: 0.0005      # Transaction cost rate (0.05%)")
            print("# dt_hours: 8.0                 # Rebalancing interval in hours")
            print("# data_dir: sample_data         # Data directory")
            print("# output_dir: backtest_results  # Output directory")
            print()
            print("# Path expansion examples:")
            print("# model_path: ~/models/model.pth              # Tilde expansion")
            print(
                "# model_path: $HOME/models/model.pth          # Environment variables"
            )
            print(
                "# model_path: ../models/model.pth             # Relative to config file"
            )
            return 0

        # Create config
        config = create_config_from_args(args)

        # Validate config
        config.validate()

        # Print config
        print("\n" + "=" * 60)
        print("BACKTEST CONFIGURATION")
        print("=" * 60)
        print(config)
        print("=" * 60 + "\n")

        # Get and print provenance info
        provenance = config.get_provenance_info()
        print("=" * 60)
        print("PROVENANCE INFO")
        print("=" * 60)
        print(f"Environment:")
        print(f"  Python: {provenance['python_version']}")
        print(f"  Platform: {provenance['platform']}")
        if config.seed is not None:
            print(f"  Seed: {config.seed}")
        if provenance["git_commit"]:
            commit_str = provenance["git_commit"][:8]  # Short hash
            dirty_marker = " (dirty)" if provenance["git_dirty"] else ""
            print(f"\nGit: {commit_str} on {provenance['git_branch']}{dirty_marker}")
        print(f"\nResolved Paths:")
        print(f"  Model: {provenance['resolved_paths']['model_path']}")
        print(f"  Data: {provenance['resolved_paths']['data_dir']}")
        print(f"  Output: {provenance['resolved_paths']['output_dir']}")
        print(f"\nTimestamp: {provenance['timestamp']}")
        print("=" * 60 + "\n")

        # Handle dry-run
        if args.dry_run:
            print("✓ Dry run complete - configuration is valid")
            print("\nFull configuration:")
            import json

            print(json.dumps(config.to_dict(), indent=2))
            return 0

        # Create backtester
        backtester = Backtester(config)

        # Run backtest
        print("Running backtest...")
        results = backtester.run(seed=config.seed)

        # Print summary using new BacktestResults methods
        use_emoji = not args.no_emoji
        if args.verbose:
            # Detailed view with all metrics
            results.print_summary(detailed=True, emoji=use_emoji)
        else:
            # Compact view for quick overview
            results.print_summary(detailed=False, emoji=use_emoji)

        # Print key insights unless disabled
        if not args.no_insights:
            results.print_key_insights(emoji=use_emoji)

        # Save results to JSON
        from pathlib import Path

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_json_path = output_dir / "results.json"
        results.to_json(
            str(results_json_path), include_raw=config.save_raw_data, indent=2
        )
        if config.save_raw_data:
            print(f"✓ Results with raw data saved to: {results_json_path}")
        else:
            print(
                f"✓ Results saved to: {results_json_path} (set save_raw_data: true in config to include positions/PnL timeseries)"
            )

        # Generate report unless disabled
        if not args.no_report:
            print(f"\nGenerating report in {config.output_dir}...")

            report_path = output_dir / "backtest_report.md"
            plot_dir = output_dir / "plots"
            report_info = results.generate_report(
                filepath=str(report_path), include_plots=True, plot_dir=str(plot_dir)
            )
            print(f"✓ Report saved to: {report_info['report_path']}")
            print(f"✓ Plots saved:")
            for plot_path in report_info["plot_paths"]:
                print(f"  - {plot_path}")

        print("\n✅ Backtest complete!")
        return 0

    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        return 130

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.verbose if hasattr(args, "verbose") else False:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
