#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto.training import TrainingConfig, Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_option_from_file(option_file: str, instrument_name: str) -> dict:
    with open(option_file, "r") as f:
        data = json.load(f)

    # Handle both formats: direct list or wrapped in 'options' key
    options = data if isinstance(data, list) else data.get("options", [])

    # Find matching instrument
    for opt in options:
        if opt["instrument_name"] == instrument_name:
            return opt

    # If not found, show available instruments
    available = [opt["instrument_name"] for opt in options]
    raise ValueError(
        f"Instrument '{instrument_name}' not found in {option_file}\n"
        f"Available instruments:\n" + "\n".join(f"  - {name}" for name in available)
    )


def create_training_config_from_option(
    option: dict,
    output_dir: str,
    epochs: int = 100,
    paths: int = 50000,
    layers: int = 4,
    units: "int | list[int]" = 128,  # Can be int or list of ints
    risk_measure: str = "expected_shortfall",
    risk_param: float = 0.9,
    transaction_cost: float = 0.0006,
    dt_hours: float = 8.0,
    volatility_override: float = None,
    volatility_window: int = 20,
    seed: int = 42,
    device: str = "cpu",
    underlying_type: str = "perpetual",
    optimizer: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 1e-6,
    features: list = None,
    model_type: str = "mlp",
) -> TrainingConfig:
    # Extract option parameters
    strike = option["strike"]
    maturity_days = option["days_to_expiry"]
    is_call = option["option_type"] == "call"
    initial_spot = option.get("initial_spot")

    # CRITICAL FIX: Normalize strike for training
    # The GBM simulator in pfhedge starts with S0=1.0. We must normalize the
    # strike to match this, otherwise log_moneyness = log(S_t / K) will be
    # completely mismatched between training (e.g., log(1/105k) ~ -11.5) and
    # backtesting (e.g., log(109k/105k) ~ +0.04).
    if not initial_spot or initial_spot <= 0:
        raise ValueError(
            f"Option data must contain 'initial_spot' for strike normalization. "
            f"Got initial_spot={initial_spot}. Cannot train without knowing the "
            f"initial spot price. Please regenerate option file with initial_spot."
        )

    normalized_strike = strike / initial_spot
    logger.info(
        f"✅ Normalizing strike for training: {strike:.2f} / {initial_spot:.2f} = {normalized_strike:.4f}"
    )

    # Use volatility override or option's IV
    if volatility_override is not None:
        volatility = volatility_override
        logger.info(f"Using volatility override: {volatility:.1%}")
    elif option.get("implied_volatility"):
        volatility = option["implied_volatility"]
        logger.info(f"Using implied volatility from premium: {volatility:.1%}")
    else:
        instrument_name = option.get("instrument_name", "UNKNOWN")
        raise ValueError(
            f"No volatility available for option {instrument_name}. "
            f"Option data missing 'implied_volatility' field. "
            f"Either provide --vol flag or regenerate option file with IV calculation."
        )

    # Create temporary config to get git hash
    temp_config = TrainingConfig(
        strike=normalized_strike,
        maturity_days=maturity_days,
        model_path="temp.pth",
    )
    provenance = temp_config.get_provenance_info()

    # Append git hash to output directory if available
    if provenance["git_commit"]:
        git_short = provenance["git_commit"][:8]
        output_dir_with_hash = f"{output_dir}_{git_short}"
        logger.info(
            f"Appending git hash to output directory: {output_dir} -> {output_dir_with_hash}"
        )
        output_dir = output_dir_with_hash

    # Create model path
    model_path = Path(output_dir) / "model.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Create config
    config = TrainingConfig(
        strike=normalized_strike,  # Use normalized strike
        maturity_days=maturity_days,
        call=is_call,
        volatility=volatility,
        volatility_window=volatility_window,
        transaction_cost=transaction_cost,
        dt_hours=dt_hours,
        underlying_type=underlying_type,
        n_paths=paths,
        n_epochs=epochs,
        model_type=model_type,
        n_layers=layers,
        n_units=units,
        risk_measure=risk_measure,
        risk_param=risk_param,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        model_path=str(model_path),
        output_dir=output_dir,
        train_seed=seed,
        test_seed=seed + 1,
        device=device,
        features=features,
    )

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train deep hedging model for specific option",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--option-file", required=True, help="JSON file from explore_options.py"
    )
    parser.add_argument(
        "--instrument",
        required=True,
        help="Instrument name to train for (e.g., BTC-29OCT24-50000-C)",
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        default="models/trained_model",
        help="Output directory for model and results (default: models/trained_model)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=50000,
        help="Number of training paths (default: 50000)",
    )
    parser.add_argument(
        "--layers", type=int, default=4, help="Number of hidden layers (default: 4)"
    )
    parser.add_argument(
        "--units",
        nargs="+",
        type=int,
        default=128,
        help="Units per layer. Can be single int (e.g., 128) or list (e.g., 64 32 32 16). Default: 128",
    )
    parser.add_argument(
        "--risk-measure",
        choices=["expected_shortfall", "variance", "cvar", "entropic"],
        default="expected_shortfall",
        help="Risk measure to optimize (default: expected_shortfall)",
    )
    parser.add_argument(
        "--risk-param",
        type=float,
        default=0.9,
        help="Risk parameter, e.g., CVaR alpha (default: 0.9)",
    )

    # Market parameters
    parser.add_argument(
        "--underlying",
        choices=["perpetual", "spot"],
        default="perpetual",
        help="Underlying instrument type (default: perpetual). "
        "Perpetual has funding rates and leverage, spot is simpler with higher costs.",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=None,
        help="Transaction cost rate. If not specified, uses default for underlying type "
        "(perpetual: 0.0006 = 0.06%%, spot: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--dt-hours",
        type=float,
        default=8.0,
        help="Time step in hours (default: 8.0, aligned with funding)",
    )

    # Overrides
    parser.add_argument(
        "--vol",
        "--volatility",
        type=float,
        dest="volatility",
        help="Override volatility (if not specified, uses IV from option)",
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=20,
        help="Rolling window for realized volatility calculation (default: 20, 0 = use constant vol)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable MLP input/output diagnostics during training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for training (default: cpu, auto=cuda if available)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=str,
        default=None,
        help="Feature list for model input (e.g., log_moneyness expiry_time volatility). "
        "Default: log_moneyness expiry_time volatility prev_hedge",
    )
    parser.add_argument(
        "--model-type",
        choices=["mlp", "lstm", "gru"],
        default="mlp",
        help="Model architecture type: mlp (feedforward), lstm (recurrent), gru (recurrent, simpler than LSTM). Default: mlp",
    )

    # Optimizer settings
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default="adamw",
        help="Optimizer: adam, adamw (default), or sgd",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1e-3,
        dest="learning_rate",
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization (default: 0.0001)",
    )

    # Early stopping
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping based on training loss",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs to wait for improvement, default: 10)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
        help="Minimum change to qualify as improvement for early stopping (default: 1e-6)",
    )

    args = parser.parse_args()

    # Determine transaction cost based on underlying type if not specified
    if args.cost is None:
        if args.underlying == "spot":
            transaction_cost = 0.001  # 0.1% for spot
        else:  # perpetual
            transaction_cost = 0.0006  # 0.06% for perpetual
    else:
        transaction_cost = args.cost

    # Determine device
    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {device}")
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"

    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Load option from file
    try:
        option = load_option_from_file(args.option_file, args.instrument)
    except FileNotFoundError:
        print(f"\nError: Option file not found: {args.option_file}")
        print("Run explore_options.py first to generate option candidates.\n")
        return 1
    except ValueError as e:
        print(f"\nError: {e}\n")
        return 1

    # Parse units argument (can be single int or list of ints)
    if isinstance(args.units, list):
        if len(args.units) == 1:
            units = args.units[0]  # Single value, convert to int
        else:
            units = args.units  # Multiple values, keep as list
    else:
        units = args.units

    # Format units for display
    if isinstance(units, list):
        units_str = f"[{', '.join(str(u) for u in units)}]"
    else:
        units_str = f"{args.layers} layers × {units} units"

    # Print option details
    print("\n" + "=" * 80)
    print("TRAINING DEEP HEDGING MODEL")
    print("=" * 80)
    print(f"\nSelected Option:")
    print(f"  Instrument: {option['instrument_name']}")
    print(f"  Type: {option['option_type'].upper()}")
    print(f"  Strike: ${option['strike']:,.2f}")
    print(f"  Initial spot: ${option['initial_spot']:,.2f}")
    print(f"  Moneyness: {option['moneyness']:.3f}")
    print(f"  Days to expiry: {option['days_to_expiry']}")
    print(f"  Premium: {option['premium_btc']:.4f} BTC (${option['premium_usd']:,.2f})")
    if option.get("implied_volatility"):
        print(f"  Implied volatility: {option['implied_volatility']:.1%}")

    print(f"\nTraining Parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Training paths: {args.paths:,}")
    print(f"  Architecture: {units_str}")
    print(f"  Risk measure: {args.risk_measure} ({args.risk_param})")
    print(f"  Underlying: {args.underlying}")
    print(f"  Transaction cost: {transaction_cost:.2%}")
    print(f"  Time step: {args.dt_hours} hours")
    print(f"  Random seed: {args.seed}")
    print(f"  Device: {device}")

    print(f"\nOutput:")
    print(f"  Directory: {args.output}")
    print("=" * 80 + "\n")

    # Calculate normalized strike for verification
    normalized_strike = option["strike"] / option["initial_spot"]

    # Create training config
    config = create_training_config_from_option(
        option=option,
        output_dir=args.output,
        epochs=args.epochs,
        paths=args.paths,
        layers=args.layers,
        units=units,  # Use parsed units (int or list)
        risk_measure=args.risk_measure,
        risk_param=args.risk_param,
        transaction_cost=transaction_cost,
        dt_hours=args.dt_hours,
        volatility_override=args.volatility,
        volatility_window=args.volatility_window,
        seed=args.seed,
        device=device,
        underlying_type=args.underlying,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        features=args.features,
        model_type=args.model_type,
    )

    # Validate config
    try:
        config.validate()
    except ValueError as e:
        print(f"\nError: Invalid configuration: {e}\n")
        return 1

    # Train model
    try:
        trainer = Trainer(config, verbose=True, enable_diagnostics=args.diagnostics)
        results = trainer.train(seed=args.seed)

        # Save training results (use config.output_dir which has git hash appended)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "training_results.json"
        results.to_json(str(results_path), include_raw=True, indent=2)

        # Save option metadata for reference
        option_path = output_dir / "option_metadata.json"
        with open(option_path, "w") as f:
            json.dump(option, f, indent=2, default=str)

        # Verify checkpoint contains normalized strike
        logger.info("Verifying saved checkpoint has normalized strike...")
        try:
            import torch

            checkpoint = torch.load(config.model_path, map_location="cpu")
            saved_strike = checkpoint.get("training_config", {}).get("strike")

            if saved_strike is None:
                logger.warning("⚠️  Checkpoint missing 'strike' in training_config")
            elif abs(saved_strike - normalized_strike) > 0.0001:
                logger.error(
                    f"❌ CHECKPOINT VERIFICATION FAILED!\n"
                    f"   Expected strike: {normalized_strike:.6f}\n"
                    f"   Saved strike: {saved_strike:.6f}\n"
                    f"   This indicates a bug in model saving!"
                )
            else:
                logger.info(
                    f"✅ Checkpoint verified: strike = {saved_strike:.6f} "
                    f"(matches normalized value)"
                )
        except Exception as e:
            logger.warning(f"⚠️  Could not verify checkpoint: {e}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\n✅ Strike Normalization Applied:")
        print(f"   Original strike: ${option['strike']:,.2f}")
        print(f"   Initial spot: ${option['initial_spot']:,.2f}")
        print(f"   Normalized strike: {normalized_strike:.6f}")
        print(f"\nFiles saved:")
        print(f"  Model: {config.model_path}")
        print(f"  Results: {results_path}")
        print(f"  Option metadata: {option_path}")
        print("\nNext step:")
        print(
            f"  Run backtest using: python -m crypto.backtest.run --model-path {config.model_path}"
        )
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
