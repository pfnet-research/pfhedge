#!/usr/bin/env python3
"""
Comprehensive loss function comparison with optimal architecture.

Tests all available loss functions with the optimal training setup:
- 10000 training paths
- 80 epochs
- 4 layers x 128 units
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
from datetime import datetime
from crypto.instruments import create_bitcoin_option_from_config
from crypto.strategies import (
    create_deep_hedger,
    compare_hedge_performance,
    calculate_bs_hedge_pnl,
)


def run_experiment(config):
    """Run a single loss function experiment."""

    # Base parameters (optimal from previous experiments)
    base_config = {
        "strike": 50000,
        "maturity_days": 14,
        "volatility": 0.8,
        "drift": 0.0,
        "cost": 0.0005,
        "train_seed": 42,
        "test_seed": 888,
        "n_paths": 10000,  # Optimal
        "n_epochs": 80,  # Optimal
        "n_layers": 4,  # Optimal
        "n_units": 128,  # Optimal
    }

    # Training
    train_config = {
        "strike": base_config["strike"],
        "maturity_days": base_config["maturity_days"],
        "call": True,
        "cost": 0.0,
        "sigma": base_config["volatility"],
        "mu": base_config["drift"],
        "underlier_cost": base_config["cost"],
        "n_paths": base_config["n_paths"],
        "seed": base_config["train_seed"],
    }

    option_train, _ = create_bitcoin_option_from_config(train_config)

    # Create and train hedger
    hedger = create_deep_hedger(
        n_layers=base_config["n_layers"],
        n_units=base_config["n_units"],
        risk_measure=config["loss"],
        risk_param=config["loss_param"],
    )

    history = hedger.fit(
        option_train,
        n_paths=base_config["n_paths"],
        n_epochs=base_config["n_epochs"],
        verbose=False,
    )

    # Test
    test_config = {
        "strike": base_config["strike"],
        "maturity_days": base_config["maturity_days"],
        "call": True,
        "cost": 0.0,
        "sigma": base_config["volatility"],
        "mu": base_config["drift"],
        "underlier_cost": base_config["cost"],
        "n_paths": 200,
        "seed": base_config["test_seed"],
    }

    option_test, _ = create_bitcoin_option_from_config(test_config)

    with torch.no_grad():
        deep_pnl = hedger.compute_cum_pl(option_test).squeeze()
        bs_delta = option_test.black_scholes_delta()
        payoffs = option_test.payoff()
        spots = option_test.underlier.spot
        bs_pnl = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, base_config["cost"])

    results = compare_hedge_performance(deep_pnl, bs_pnl)

    improvement = results["Deep Hedge"]["mean"] - results["Black-Scholes"]["mean"]
    beats_bs = improvement > 0

    return {
        "name": config["name"],
        "config": config,
        "deep": results["Deep Hedge"],
        "bs": results["Black-Scholes"],
        "improvement": improvement,
        "beats_bs": beats_bs,
        "training_improvement": (
            (history[0] - history[-1]) / history[0] * 100 if len(history) > 0 else 0
        ),
    }


def main():
    """Run comprehensive loss function optimization."""

    print("=" * 70)
    print("LOSS FUNCTION OPTIMIZATION WITH OPTIMAL ARCHITECTURE")
    print("=" * 70)
    print("Architecture: 10000 paths, 80 epochs, 4 layers x 128 units")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    experiments = [
        # ExpectedShortfall with different quantiles
        {
            "name": "ES (p=0.1) - Extreme tail",
            "loss": "expected_shortfall",
            "loss_param": 0.1,
        },
        {
            "name": "ES (p=0.3) - Upper tail",
            "loss": "expected_shortfall",
            "loss_param": 0.3,
        },
        {
            "name": "ES (p=0.5) - Median (CURRENT BEST)",
            "loss": "expected_shortfall",
            "loss_param": 0.5,
        },
        {
            "name": "ES (p=0.7) - Lower tail",
            "loss": "expected_shortfall",
            "loss_param": 0.7,
        },
        {
            "name": "ES (p=0.9) - Near mean",
            "loss": "expected_shortfall",
            "loss_param": 0.9,
        },
        # EntropicRiskMeasure with different risk aversions
        {
            "name": "Entropic (a=0.1) - Low aversion",
            "loss": "entropic",
            "loss_param": 0.1,
        },
        {
            "name": "Entropic (a=0.5) - Medium-low",
            "loss": "entropic",
            "loss_param": 0.5,
        },
        {"name": "Entropic (a=1.0) - Medium", "loss": "entropic", "loss_param": 1.0},
        {
            "name": "Entropic (a=2.0) - High aversion",
            "loss": "entropic",
            "loss_param": 2.0,
        },
        # EntropicLoss (expected exponential utility)
        {"name": "EntropicLoss (a=0.1)", "loss": "entropic_loss", "loss_param": 0.1},
        {"name": "EntropicLoss (a=0.5)", "loss": "entropic_loss", "loss_param": 0.5},
        {"name": "EntropicLoss (a=1.0)", "loss": "entropic_loss", "loss_param": 1.0},
        {"name": "EntropicLoss (a=2.0)", "loss": "entropic_loss", "loss_param": 2.0},
        # QuadraticCVaR (Buehler 2019)
        {
            "name": "QuadCVaR (lam=1.5) - Mild",
            "loss": "quadratic_cvar",
            "loss_param": 1.5,
        },
        {
            "name": "QuadCVaR (lam=2.0) - Medium",
            "loss": "quadratic_cvar",
            "loss_param": 2.0,
        },
        {
            "name": "QuadCVaR (lam=5.0) - Strong",
            "loss": "quadratic_cvar",
            "loss_param": 5.0,
        },
        {
            "name": "QuadCVaR (lam=10.0) - Very strong",
            "loss": "quadratic_cvar",
            "loss_param": 10.0,
        },
    ]

    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Testing: {exp['name']}")
        print(f"  Loss: {exp['loss']}({exp['loss_param']})")

        try:
            result = run_experiment(exp)
            results.append(result)

            status = "✅ WIN" if result["beats_bs"] else "❌ LOSS"
            print(
                f"  Result: Deep ${result['deep']['mean']:.0f} vs BS ${result['bs']['mean']:.0f} ({result['improvement']:+.0f}) {status}"
            )
            print(f"  Training improvement: {result['training_improvement']:.1f}%")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n\n" + "=" * 70)
    print("LOSS FUNCTION OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\n{'Configuration':<40} {'Improv':>10} {'Status':>8}")
    print("-" * 70)

    for r in results:
        name = r["name"][:38]
        improvement = r["improvement"]
        status = "✅" if r["beats_bs"] else "❌"

        print(f"{name:<40} ${improvement:>9.0f} {status:>8}")

    # Find best
    if results:
        wins = [r for r in results if r["beats_bs"]]
        if wins:
            best = max(wins, key=lambda r: r["improvement"])
            print(f"\n{'='*70}")
            print(f"🏆 BEST LOSS FUNCTION: {best['name']}")
            print(f"{'='*70}")
            print(f"Deep Hedge PnL: ${best['deep']['mean']:,.2f}")
            print(f"BS PnL: ${best['bs']['mean']:,.2f}")
            print(f"Improvement: ${best['improvement']:,.2f}")
            print(f"Deep Sharpe: {best['deep']['sharpe']:.3f}")
            print(f"BS Sharpe: {best['bs']['sharpe']:.3f}")
            print(f"Training improvement: {best['training_improvement']:.1f}%")
            print(
                f"\nLoss function: {best['config']['loss']}(param={best['config']['loss_param']})"
            )

            # Compare with current best (ES p=0.5)
            current_best = next(
                (
                    r
                    for r in results
                    if r["name"] == "ES (p=0.5) - Median (CURRENT BEST)"
                ),
                None,
            )
            if current_best and best["name"] != current_best["name"]:
                diff = best["improvement"] - current_best["improvement"]
                print(f"\n💡 This is ${diff:.2f} better than current best (ES p=0.5)")
        else:
            print("\n⚠️  NO CONFIGURATIONS BEAT BLACK-SCHOLES")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save
    import json

    with open("crypto/utils/loss_function_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = main()
