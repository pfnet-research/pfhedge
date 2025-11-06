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


def test_loss_function(loss_config):
    print(f"\n{'='*70}")
    print(f"Testing: {loss_config['name']}")
    print(f"{'='*70}")
    print(f"Loss: {loss_config['risk_measure']}, Param: {loss_config['risk_param']}")

    # Base configuration (good training setup)
    base_config = {
        "strike": 50000,
        "maturity_days": 14,
        "volatility": 0.8,
        "drift": 0.0,
        "cost": 0.0005,  # 0.05% from Deribit
        "train_seed": 42,
        "test_seed": 888,
    }

    # Training configuration
    train_config = {
        **base_config,
        "call": True,
        "cost": 0.0,
        "sigma": base_config["volatility"],
        "mu": base_config["drift"],
        "underlier_cost": base_config["cost"],
        "n_paths": loss_config.get("n_paths", 2000),
        "seed": base_config["train_seed"],
    }

    print(
        f"Training with {train_config['n_paths']} paths for {loss_config.get('n_epochs', 40)} epochs..."
    )

    # Create training option
    option_train, _ = create_bitcoin_option_from_config(train_config)

    # Create hedger with specific loss function
    try:
        hedger = create_deep_hedger(
            n_layers=loss_config.get("n_layers", 3),
            n_units=loss_config.get("n_units", 64),
            risk_measure=loss_config["risk_measure"],
            risk_param=loss_config["risk_param"],
        )
    except Exception as e:
        print(f"❌ Failed to create hedger: {e}")
        return None

    # Train
    try:
        history = hedger.fit(
            option_train,
            n_paths=train_config["n_paths"],
            n_epochs=loss_config.get("n_epochs", 40),
            verbose=False,
        )

        initial_loss = history[0] if len(history) > 0 else 0
        final_loss = history[-1] if len(history) > 0 else 0
        improvement = (
            (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
        )

        print(
            f"Training: {initial_loss:.2f} → {final_loss:.2f} ({improvement:.1f}% improvement)"
        )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Test configuration
    test_config = {
        **base_config,
        "call": True,
        "cost": 0.0,
        "sigma": base_config["volatility"],
        "mu": base_config["drift"],
        "underlier_cost": base_config["cost"],
        "n_paths": 100,  # More test paths for better evaluation
        "seed": base_config["test_seed"],
    }

    option_test, _ = create_bitcoin_option_from_config(test_config)

    # Compute strategies
    with torch.no_grad():
        deep_hedge_pnl = hedger.compute_cum_pl(option_test).squeeze()
        bs_delta = option_test.black_scholes_delta()
        payoffs = option_test.payoff()
        spots = option_test.underlier.spot
        bs_hedge_pnl = calculate_bs_hedge_pnl(
            spots, bs_delta, payoffs, base_config["cost"]
        )

    # Compare
    results = compare_hedge_performance(deep_hedge_pnl, bs_hedge_pnl)

    deep_mean = results["Deep Hedge"]["mean"]
    deep_std = results["Deep Hedge"]["std"]
    deep_sharpe = results["Deep Hedge"]["sharpe"]

    bs_mean = results["Black-Scholes"]["mean"]
    bs_std = results["Black-Scholes"]["std"]
    bs_sharpe = results["Black-Scholes"]["sharpe"]

    # Calculate improvements
    mean_improvement = (deep_mean - bs_mean) / abs(bs_mean) * 100 if bs_mean != 0 else 0
    std_improvement = (bs_std - deep_std) / bs_std * 100 if bs_std != 0 else 0
    sharpe_improvement = (
        (deep_sharpe - bs_sharpe) / abs(bs_sharpe) * 100 if bs_sharpe != 0 else 0
    )

    # Success if deep hedge beats BS in Sharpe ratio
    success = deep_sharpe > bs_sharpe

    print(f"\nResults:")
    print(
        f"  Deep:  ${deep_mean:>8.2f} ± ${deep_std:>8.2f} (Sharpe: {deep_sharpe:>6.3f})"
    )
    print(f"  BS:    ${bs_mean:>8.2f} ± ${bs_std:>8.2f} (Sharpe: {bs_sharpe:>6.3f})")
    print(
        f"  {'✅' if success else '❌'} Sharpe improvement: {sharpe_improvement:+.1f}%"
    )

    return {
        "name": loss_config["name"],
        "config": loss_config,
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
        "success": success,
    }


def run_loss_function_experiments():

    print("=" * 70)
    print("LOSS FUNCTION COMPARISON FOR BITCOIN DEEP HEDGING")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define loss function configurations to test
    loss_configs = [
        # Baseline
        {
            "name": "ExpectedShortfall (p=0.5)",
            "risk_measure": "expected_shortfall",
            "risk_param": 0.5,
            "n_epochs": 40,
            "n_paths": 2000,
        },
        # Focus on tail risk
        {
            "name": "ExpectedShortfall (p=0.1) - Tail",
            "risk_measure": "expected_shortfall",
            "risk_param": 0.1,
            "n_epochs": 40,
            "n_paths": 2000,
        },
        # Entropic risk measure (moderate risk aversion)
        {
            "name": "EntropicRiskMeasure (a=1.0)",
            "risk_measure": "entropic",
            "risk_param": 1.0,
            "n_epochs": 40,
            "n_paths": 2000,
        },
        # Entropic risk measure (higher risk aversion)
        {
            "name": "EntropicRiskMeasure (a=0.1)",
            "risk_measure": "entropic",
            "risk_param": 0.1,
            "n_epochs": 40,
            "n_paths": 2000,
        },
        # Entropic loss (expected utility)
        {
            "name": "EntropicLoss (a=1.0)",
            "risk_measure": "entropic_loss",
            "risk_param": 1.0,
            "n_epochs": 40,
            "n_paths": 2000,
        },
        # Quadratic CVaR (Buehler)
        {
            "name": "QuadraticCVaR (lam=2.0)",
            "risk_measure": "quadratic_cvar",
            "risk_param": 2.0,
            "n_epochs": 40,
            "n_paths": 2000,
        },
        # Quadratic CVaR (higher penalty)
        {
            "name": "QuadraticCVaR (lam=10.0)",
            "risk_measure": "quadratic_cvar",
            "risk_param": 10.0,
            "n_epochs": 40,
            "n_paths": 2000,
        },
    ]

    results = []

    for i, config in enumerate(loss_configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"# Test {i}/{len(loss_configs)}")
        print(f"{'#'*70}")

        try:
            result = test_loss_function(config)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n\n" + "=" * 70)
    print("LOSS FUNCTION COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Loss Function':<40} {'Sharpe':>10} {'vs BS':>12} {'Status':>8}")
    print("-" * 70)

    for result in results:
        name = result["name"][:38]
        deep_sharpe = result["deep_hedge"]["sharpe"]
        sharpe_imp = result["improvements"]["sharpe"]
        status = "✅ WIN" if result["success"] else "❌ LOSS"

        print(f"{name:<40} {deep_sharpe:>10.3f} {sharpe_imp:>10.1f}% {status:>8}")

    # Find best
    if results:
        best_result = max(results, key=lambda r: r["deep_hedge"]["sharpe"])
        print(f"\n{'='*70}")
        print(f"🏆 BEST LOSS FUNCTION: {best_result['name']}")
        print(f"{'='*70}")
        print(f"Sharpe Ratio: {best_result['deep_hedge']['sharpe']:.3f}")
        print(f"Mean PnL: ${best_result['deep_hedge']['mean']:,.2f}")
        print(f"Std PnL: ${best_result['deep_hedge']['std']:,.2f}")
        print(f"Improvement over BS: {best_result['improvements']['sharpe']:+.1f}%")

        # Save results
        import json

        output_file = os.path.join(
            os.path.dirname(__file__), "loss_function_results.json"
        )

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to {output_file}")
    else:
        print("\n❌ No successful results")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    results = run_loss_function_experiments()
