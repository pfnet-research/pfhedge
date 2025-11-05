#!/usr/bin/env python3
"""Train deep hedging model with SPOT underlying (not perpetual).

Uses same parameters as Phase 4 best model but with spot instead of perpetual.
"""

import argparse
import json
import logging
import time
from pathlib import Path
import torch
import numpy as np

from crypto.instruments import BitcoinSpotBrownian, BitcoinEuropeanOption
from crypto.strategies import create_deep_hedger, calculate_bs_hedge_pnl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_spot_model(
    n_paths: int = 50000,
    n_epochs: int = 50,
    seed: int = 42,
    output_dir: str = "results/spot_model",
):
    """Train model with spot underlying using Phase 4 optimal config."""

    logger.info(f"\n{'='*60}")
    logger.info("Training Deep Hedging Model with SPOT Underlying")
    logger.info(f"{'='*60}")
    logger.info(f"Paths: {n_paths:,}")
    logger.info(f"Epochs: {n_epochs}")
    logger.info(f"Seed: {seed}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create SPOT underlier (not perpetual)
    logger.info("\nCreating SPOT underlier...")
    underlier = BitcoinSpotBrownian(
        dt=8 / 24 / 365,
        sigma=0.8,
        mu=0.0,
        cost=0.001,  # 10 bps transaction cost
        volatility_window=20,
    )

    # Create option
    logger.info("Creating European call option...")
    option = BitcoinEuropeanOption(
        underlier=underlier,
        strike=1.0,  # Normalized
        maturity=14 / 365,  # 14 days
        call=True,
    )

    # Simulate paths
    logger.info(f"Simulating {n_paths:,} paths...")
    option.simulate(n_paths=n_paths, init_state=(1.0,))
    logger.info(f"Simulation complete. Shape: {option.underlier.spot.shape}")

    # Create hedger with optimal config from Phase 4
    logger.info("\nCreating hedger (MLP + Entropic)...")
    hedger = create_deep_hedger(
        model_type="mlp",
        n_layers=2,
        n_units=32,
        risk_measure="entropic",
        risk_param=2.0,
    )

    # Train
    from torch.optim import Adam

    logger.info(f"\nTraining for {n_epochs} epochs...")

    train_start = time.time()
    loss_history = hedger.fit(
        option,
        n_epochs=n_epochs,
        n_paths=n_paths,
        optimizer=Adam,
        verbose=True,
        validation=False,
    )
    train_time = time.time() - train_start

    logger.info(f"\nTraining complete in {train_time:.1f}s ({train_time/60:.1f}min)")

    # Evaluate
    logger.info("\nEvaluating model...")
    hedge_positions = hedger.compute_hedge(option)

    if hedge_positions.dim() == 3:
        hedge_positions = hedge_positions.squeeze(1)

    # Calculate PnL
    spots = option.underlier.spot
    payoffs = option.payoff()

    pnl = calculate_bs_hedge_pnl(
        spots=spots,
        bs_delta=hedge_positions,
        payoffs=payoffs,
        cost=0.001,
    )

    final_pnl = pnl[:, -1]

    # Metrics
    metrics = {
        "underlying_type": "spot",
        "n_paths": n_paths,
        "n_epochs": n_epochs,
        "seed": seed,
        "train_time_sec": train_time,
        "train_time_min": train_time / 60,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "pnl_mean": final_pnl.mean().item(),
        "pnl_std": final_pnl.std().item(),
        "pnl_min": final_pnl.min().item(),
        "pnl_max": final_pnl.max().item(),
        "pnl_q05": final_pnl.quantile(0.05).item(),
        "pnl_q25": final_pnl.quantile(0.25).item(),
        "pnl_q50": final_pnl.quantile(0.50).item(),
        "pnl_q75": final_pnl.quantile(0.75).item(),
        "pnl_q95": final_pnl.quantile(0.95).item(),
        "sharpe": final_pnl.mean().item() / (final_pnl.std().item() + 1e-8),
        "n_params": sum(p.numel() for p in hedger.model.parameters()),
    }

    logger.info(f"\n{'='*60}")
    logger.info("Training Results:")
    logger.info(f"{'='*60}")
    logger.info(f"  PnL mean: {metrics['pnl_mean']:.6f}")
    logger.info(f"  PnL std: {metrics['pnl_std']:.6f}")
    logger.info(f"  Sharpe: {metrics['sharpe']:.6f}")
    logger.info(f"  Training time: {metrics['train_time_min']:.1f} min")
    logger.info(f"  Parameters: {metrics['n_params']:,}")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save with backtest-compatible format
    checkpoint = {
        "model_state_dict": {},
        "model_config": {
            "model_type": "mlp",
            "n_layers": 2,
            "n_units": 32,
            "risk_measure": "entropic",
            "risk_param": 2.0,
            "features": ["log_moneyness", "expiry_time", "volatility", "prev_hedge"],
            "underlying_type": "spot",  # Important!
        },
        "criterion_config": {
            "risk_measure": "entropic",
            "risk_param": 2.0,
        },
        "metrics": metrics,
    }

    # Fix state_dict keys - add 'model.' prefix for backtest compatibility
    for key, value in hedger.model.state_dict().items():
        checkpoint["model_state_dict"][f"model.{key}"] = value

    model_path = output_path / "hedger_spot.pth"
    torch.save(checkpoint, model_path)
    logger.info(f"\n✅ Model saved to: {model_path}")

    # Save metrics
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✅ Metrics saved to: {metrics_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}\n")

    return metrics, hedger


def main():
    parser = argparse.ArgumentParser(description="Train spot hedging model")
    parser.add_argument("--n-paths", type=int, default=50000, help="Number of paths")
    parser.add_argument("--n-epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="results/spot_model", help="Output directory"
    )

    args = parser.parse_args()

    metrics, hedger = train_spot_model(
        n_paths=args.n_paths,
        n_epochs=args.n_epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    return metrics


if __name__ == "__main__":
    main()
