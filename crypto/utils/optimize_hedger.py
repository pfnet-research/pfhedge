#!/usr/bin/env python3

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
from pfhedge.nn import EntropicRiskMeasure, ExpectedShortfall, QuadraticCVaR


def run_experiment(config):
    experiment_name = config["name"]
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    print(f"Loss: {config['loss_name']}, Risk param: {config['risk_param']}")
    print(f"Architecture: {config['n_layers']} layers x {config['n_units']} units")
    print(f"Training: {config['n_epochs']} epochs x {config['n_paths']} paths")
    print(f"Transaction cost: {config['cost']:.3%}")

    # Create training option
    train_config = {
        "strike": config["strike"],
        "maturity_days": config["maturity_days"],
        "call": True,
        "cost": 0.0,
        "sigma": config["volatility"],
        "mu": config["drift"],
        "underlier_cost": config["cost"],
        "n_paths": config["n_paths"],
        "seed": config["train_seed"],
    }
    option_train, _ = create_bitcoin_option_from_config(train_config)

    # Create hedger
    hedger = create_deep_hedger(
        n_layers=config["n_layers"],
        n_units=config["n_units"],
        risk_measure=config["loss_name"],
        risk_param=config["risk_param"],
    )

    # Train
    print(f"\nTraining...")
    history = hedger.fit(
        option_train,
        n_paths=config["n_paths"],
        n_epochs=config["n_epochs"],
        verbose=False,
    )

    initial_loss = history[0] if len(history) > 0 else 0
    final_loss = history[-1] if len(history) > 0 else 0
    improvement = (
        (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
    )

    print(
        f"Training complete: {initial_loss:.2f} → {final_loss:.2f} ({improvement:.1f}% improvement)"
    )

    # Test
    test_config = {
        "strike": config["strike"],
        "maturity_days": config["maturity_days"],
        "call": True,
        "cost": 0.0,
        "sigma": config["volatility"],
        "mu": config["drift"],
        "underlier_cost": config["cost"],
        "n_paths": config["test_n_paths"],
        "seed": config["test_seed"],
    }
    option_test, _ = create_bitcoin_option_from_config(test_config)

    # Compute hedge strategies
    with torch.no_grad():
        deep_hedge_pnl = hedger.compute_cum_pl(option_test).squeeze()
        bs_delta = option_test.black_scholes_delta()
        payoffs = option_test.payoff()
        spots = option_test.underlier.spot
        bs_hedge_pnl = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, config["cost"])

    # Compare performance
    results = compare_hedge_performance(deep_hedge_pnl, bs_hedge_pnl)

    # Extract metrics
    deep_mean = results["Deep Hedge"]["mean"]
    deep_std = results["Deep Hedge"]["std"]
    deep_sharpe = results["Deep Hedge"]["sharpe"]

    bs_mean = results["Black-Scholes"]["mean"]
    bs_std = results["Black-Scholes"]["std"]
    bs_sharpe = results["Black-Scholes"]["sharpe"]

    # Compute improvement over BS
    mean_improvement = (deep_mean - bs_mean) / abs(bs_mean) * 100 if bs_mean != 0 else 0
    std_improvement = (bs_std - deep_std) / bs_std * 100 if bs_std != 0 else 0
    sharpe_improvement = (
        (deep_sharpe - bs_sharpe) / abs(bs_sharpe) * 100 if bs_sharpe != 0 else 0
    )

    print(f"\nResults:")
    print(
        f"  Deep PnL: ${deep_mean:,.2f} ± ${deep_std:,.2f} (Sharpe: {deep_sharpe:.3f})"
    )
    print(f"  BS PnL:   ${bs_mean:,.2f} ± ${bs_std:,.2f} (Sharpe: {bs_sharpe:.3f})")
    print(
        f"  Improvements: Mean {mean_improvement:+.1f}%, Risk {std_improvement:+.1f}%, Sharpe {sharpe_improvement:+.1f}%"
    )

    return {
        "name": experiment_name,
        "config": config,
        "training": {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improvement": improvement,
        },
        "deep_hedge": results["Deep Hedge"],
        "black_scholes": results["Black-Scholes"],
        "improvements": {
            "mean": mean_improvement,
            "std": std_improvement,
            "sharpe": sharpe_improvement,
        },
    }


def run_optimization_suite():

    print("=" * 70)
    print("BITCOIN DEEP HEDGING OPTIMIZATION SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Base configuration
    base_config = {
        "strike": 50000,
        "maturity_days": 14,
        "volatility": 0.8,
        "drift": 0.0,
        "cost": 0.0005,  # 0.05% from Deribit
        "train_seed": 42,
        "test_seed": 888,
        "test_n_paths": 100,  # More test paths for better evaluation
    }

    experiments = []

    # Experiment 1: Baseline with fixed cost
    experiments.append(
        {
            **base_config,
            "name": "Baseline (fixed cost)",
            "loss_name": "expected_shortfall",
            "risk_param": 0.5,
            "n_layers": 3,
            "n_units": 64,
            "n_paths": 1000,
            "n_epochs": 20,
        }
    )

    # Experiment 2: More epochs
    experiments.append(
        {
            **base_config,
            "name": "More epochs (50)",
            "loss_name": "expected_shortfall",
            "risk_param": 0.5,
            "n_layers": 3,
            "n_units": 64,
            "n_paths": 1000,
            "n_epochs": 50,
        }
    )

    # Experiment 3: More paths
    experiments.append(
        {
            **base_config,
            "name": "More paths (3000)",
            "loss_name": "expected_shortfall",
            "risk_param": 0.5,
            "n_layers": 3,
            "n_units": 64,
            "n_paths": 3000,
            "n_epochs": 30,
        }
    )

    # Experiment 4: Deeper network
    experiments.append(
        {
            **base_config,
            "name": "Deeper network (5 layers)",
            "loss_name": "expected_shortfall",
            "risk_param": 0.5,
            "n_layers": 5,
            "n_units": 64,
            "n_paths": 2000,
            "n_epochs": 40,
        }
    )

    # Experiment 5: Wider network
    experiments.append(
        {
            **base_config,
            "name": "Wider network (128 units)",
            "loss_name": "expected_shortfall",
            "risk_param": 0.5,
            "n_layers": 3,
            "n_units": 128,
            "n_paths": 2000,
            "n_epochs": 40,
        }
    )

    # Experiment 6: Different risk parameter (focus on tail)
    experiments.append(
        {
            **base_config,
            "name": "Tail risk (p=0.1)",
            "loss_name": "expected_shortfall",
            "risk_param": 0.1,
            "n_layers": 3,
            "n_units": 64,
            "n_paths": 2000,
            "n_epochs": 40,
        }
    )

    # Experiment 7: Best combination
    experiments.append(
        {
            **base_config,
            "name": "Best combination",
            "loss_name": "expected_shortfall",
            "risk_param": 0.1,
            "n_layers": 4,
            "n_units": 128,
            "n_paths": 3000,
            "n_epochs": 60,
        }
    )

    # Run all experiments
    results = []
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Experiment {i}/{len(experiments)}")
        print(f"{'#'*70}")

        try:
            result = run_experiment(exp_config)
            results.append(result)
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Experiment':<30} {'Mean PnL':>12} {'Sharpe':>10} {'vs BS':>12}")
    print("-" * 70)

    for result in results:
        name = result["name"][:28]
        deep_mean = result["deep_hedge"]["mean"]
        deep_sharpe = result["deep_hedge"]["sharpe"]
        mean_imp = result["improvements"]["mean"]

        indicator = (
            "✅"
            if mean_imp > 0 and deep_sharpe > result["black_scholes"]["sharpe"]
            else "⚠️ "
        )

        print(
            f"{indicator} {name:<28} ${deep_mean:>10.2f} {deep_sharpe:>10.3f} {mean_imp:>10.1f}%"
        )

    # Find best
    best_result = max(results, key=lambda r: r["deep_hedge"]["sharpe"])
    print(f"\n{'='*70}")
    print(f"🏆 BEST CONFIGURATION: {best_result['name']}")
    print(f"{'='*70}")
    print(f"Sharpe Ratio: {best_result['deep_hedge']['sharpe']:.3f}")
    print(f"Mean PnL: ${best_result['deep_hedge']['mean']:,.2f}")
    print(f"Std PnL: ${best_result['deep_hedge']['std']:,.2f}")

    # Save results
    import json

    output_file = os.path.join(os.path.dirname(__file__), "optimization_results.json")

    # Convert to JSON-serializable format
    results_json = []
    for r in results:
        r_copy = r.copy()
        # Convert torch tensors if any
        results_json.append(r_copy)

    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    results = run_optimization_suite()
