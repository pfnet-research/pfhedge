#!/usr/bin/env python3
"""
Bitcoin Deep Hedging Model Training Script

Simple CLI script to train a deep hedging model with customizable parameters.
This script uses the training framework to train and save a model.

Usage:
    python train_model.py                           # Default parameters
    python train_model.py --epochs 100              # Custom epochs
    python train_model.py --strike 60000 --vol 1.0  # Custom strike and volatility
    python train_model.py --help                    # Show all options
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from crypto.training import TrainingConfig, Trainer


def main():
    """Run model training with command-line arguments."""

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

    # ========== Early Device Validation ==========

    # Check if CUDA is requested but not available
    if "cuda" in args.device.lower():
        import torch

        if not torch.cuda.is_available():
            print("\n" + "=" * 70)
            print("❌ CUDA ERROR")
            print("=" * 70)
            print(
                f"\nCUDA device requested (--device {args.device}), but CUDA is not available."
            )
            print("\nPossible causes:")
            print("  • PyTorch not installed with CUDA support")
            print("  • No NVIDIA GPU detected")
            print("  • CUDA drivers not properly installed")
            print("\nSuggestions:")
            print("  • Use --device cpu for CPU training")
            print("  • Reinstall PyTorch with CUDA support: https://pytorch.org/")
            print("  • Check GPU availability: nvidia-smi")
            print("=" * 70 + "\n")
            return 1

    # ========== Create Configuration ==========

    print("\n" + "=" * 70)
    print("BITCOIN DEEP HEDGING - MODEL TRAINING")
    print("=" * 70)

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

    print(f"\nConfiguration:")
    print(f"  Option: {'Call' if config.call else 'Put'} @ ${config.strike:,.0f}")
    print(f"  Maturity: {config.maturity_days} days")
    print(f"  Volatility: {config.volatility:.1%}")
    print(f"  Transaction cost: {config.transaction_cost:.2%}")
    print(f"  Time step: {config.dt_hours} hours")
    print(f"\nTraining:")
    print(f"  Paths: {config.n_paths:,}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Seed: {config.train_seed}")
    print(f"\nModel:")
    print(f"  Architecture: {config.n_layers} layers × {config.n_units} units")
    print(f"  Risk measure: {config.risk_measure} (param={config.risk_param})")
    print(f"  Device: {config.device}")
    print(f"\nOutput:")
    print(f"  Model: {config.model_path}")
    print(f"  Results: {config.output_dir}/")
    print("=" * 70)

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return 1

    # ========== Train Model ==========

    # Create trainer
    trainer = Trainer(config)

    # Run training
    try:
        results = trainer.train(seed=args.seed)

        # Export results
        os.makedirs(config.output_dir, exist_ok=True)
        results_file = os.path.join(config.output_dir, "training_results.json")
        results.to_json(results_file, include_raw=True, indent=2)

        print(f"\n✅ Training results saved to: {results_file}")

        # Print final summary
        summary = results.summary()
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"\nFinal Training Loss: {summary['final_loss']:.6f}")
        print(f"Improvement: {summary['improvement_pct']:.1f}%")

        test_metrics = summary["test_metrics"]
        deep = test_metrics["deep_hedge"]
        bs = test_metrics["bs_baseline"]

        print(f"\nTest Performance:")
        print(
            f"  Deep hedge: ${deep['mean_pnl']:,.2f} ± ${deep['std_pnl']:,.2f} (Sharpe: {deep['sharpe_ratio']:.3f})"
        )
        print(
            f"  BS baseline: ${bs['mean_pnl']:,.2f} ± ${bs['std_pnl']:,.2f} (Sharpe: {bs['sharpe_ratio']:.3f})"
        )

        improvement = test_metrics["comparison"]["mean_pnl_improvement"]
        sharpe_improvement = test_metrics["comparison"]["sharpe_improvement"]
        print(
            f"  Improvement: ${improvement:+,.2f} (Sharpe: {sharpe_improvement:+.3f})"
        )

        print(f"\nModel saved to: {results.model_path}")
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
