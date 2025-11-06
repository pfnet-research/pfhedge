#!/usr/bin/env python3

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from crypto.training import TrainingConfig, Trainer


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a deep hedging model for Bitcoin options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Option parameters
    parser.add_argument(
        "--strike",
        type=float,
        default=50000,
        help="Option strike price",
    )
    parser.add_argument(
        "--maturity",
        type=int,
        default=14,
        help="Option maturity in days",
    )
    parser.add_argument(
        "--call",
        action="store_true",
        default=True,
        help="Call option (default: True)",
    )
    parser.add_argument(
        "--put",
        action="store_true",
        help="Put option (overrides --call)",
    )

    # Market parameters
    parser.add_argument(
        "--vol",
        "--volatility",
        type=float,
        default=0.8,
        dest="volatility",
        help="Volatility for simulation (e.g., 0.8 for 80%%)",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=0.0,
        help="Drift for simulation",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.0005,
        help="Transaction cost rate (e.g., 0.0005 for 0.05%%)",
    )
    parser.add_argument(
        "--dt-hours",
        type=float,
        default=8.0,
        help="Time step in hours",
    )

    # Training parameters
    parser.add_argument(
        "--paths",
        type=int,
        default=10000,
        help="Number of training paths",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Model architecture
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--units",
        type=int,
        default=128,
        help="Number of units per layer",
    )
    parser.add_argument(
        "--risk-measure",
        type=str,
        default="expected_shortfall",
        choices=["expected_shortfall", "variance", "cvar", "entropic"],
        help="Risk measure for training",
    )
    parser.add_argument(
        "--risk-param",
        type=float,
        default=0.9,
        help="Risk parameter (e.g., CVaR alpha)",
    )

    # Output
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/deep_hedger_trained.pth",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_results",
        help="Directory for training outputs",
    )

    # Test parameters
    parser.add_argument(
        "--test-paths",
        type=int,
        default=200,
        help="Number of test paths for evaluation",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for training (cpu, cuda, cuda:0, etc.)",
    )

    args = parser.parse_args()

    # ========== Create Configuration ==========

    # Determine call/put
    is_call = not args.put  # Put overrides call

    # Create configuration
    config = TrainingConfig(
        strike=args.strike,
        maturity_days=args.maturity,
        call=is_call,
        volatility=args.volatility,
        drift=args.drift,
        transaction_cost=args.cost,
        dt_hours=args.dt_hours,
        n_paths=args.paths,
        n_epochs=args.epochs,
        n_layers=args.layers,
        n_units=args.units,
        risk_measure=args.risk_measure,
        risk_param=args.risk_param,
        model_path=args.model_path,
        test_n_paths=args.test_paths,
        test_seed=args.seed + 1,  # Different seed for test
        train_seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print("\n" + "=" * 70)
        print("❌ CONFIGURATION ERROR")
        print("=" * 70)
        print(f"\n{e}")
        print("=" * 70 + "\n")
        return 1

    # ========== Train Model ==========

    # Create trainer with verbose=True for CLI output
    # (CUDA availability checked here)
    try:
        trainer = Trainer(config, verbose=True)
    except ValueError as e:
        print("\n" + "=" * 70)
        print("❌ DEVICE ERROR")
        print("=" * 70)
        print(f"\n{e}")
        print("=" * 70 + "\n")
        return 1

    # Run training
    try:
        results = trainer.train(seed=args.seed)

        # Export results
        os.makedirs(config.output_dir, exist_ok=True)
        results_file = os.path.join(config.output_dir, "training_results.json")
        results.to_json(results_file, include_raw=True, indent=2)

        print(f"✅ Training results saved to: {results_file}\n")

        return 0

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
