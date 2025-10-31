#!/usr/bin/env python3
"""
Robustness testing for deep hedging across different market conditions.

Tests the optimal configuration (10k paths, 80 epochs, 4x128, ES p=0.9) on:
- Different strike prices (ITM, ATM, OTM)
- Different maturities (7, 14, 21, 30 days)
- Different volatilities (0.4, 0.8, 1.2)
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


def run_experiment(config, scenario):
    """Run a single robustness experiment."""

    # Training
    train_config = {
        "strike": scenario["strike"],
        "maturity_days": scenario["maturity_days"],
        "call": True,
        "cost": 0.0,
        "sigma": scenario["volatility"],
        "mu": scenario["drift"],
        "underlier_cost": scenario["cost"],
        "n_paths": config["n_paths"],
        "seed": config["train_seed"],
    }

    option_train, _ = create_bitcoin_option_from_config(train_config)

    # Create and train hedger with optimal settings
    hedger = create_deep_hedger(
        n_layers=config["n_layers"],
        n_units=config["n_units"],
        risk_measure=config["loss"],
        risk_param=config["loss_param"],
    )

    history = hedger.fit(
        option_train,
        n_paths=config["n_paths"],
        n_epochs=config["n_epochs"],
        verbose=False,
    )

    # Test
    test_config = {
        "strike": scenario["strike"],
        "maturity_days": scenario["maturity_days"],
        "call": True,
        "cost": 0.0,
        "sigma": scenario["volatility"],
        "mu": scenario["drift"],
        "underlier_cost": scenario["cost"],
        "n_paths": 200,
        "seed": config["test_seed"],
    }

    option_test, _ = create_bitcoin_option_from_config(test_config)

    with torch.no_grad():
        deep_pnl = hedger.compute_cum_pl(option_test).squeeze()
        bs_delta = option_test.black_scholes_delta()
        payoffs = option_test.payoff()
        spots = option_test.underlier.spot
        bs_pnl = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, scenario["cost"])

    results = compare_hedge_performance(deep_pnl, bs_pnl)

    improvement = results["Deep Hedge"]["mean"] - results["Black-Scholes"]["mean"]
    beats_bs = improvement > 0

    # Calculate relative improvement percentage
    rel_improvement = (
        (improvement / abs(results["Black-Scholes"]["mean"])) * 100
        if results["Black-Scholes"]["mean"] != 0
        else 0
    )

    return {
        "scenario": scenario,
        "deep": results["Deep Hedge"],
        "bs": results["Black-Scholes"],
        "improvement": improvement,
        "rel_improvement": rel_improvement,
        "beats_bs": beats_bs,
        "training_improvement": (
            (history[0] - history[-1]) / history[0] * 100 if len(history) > 0 else 0
        ),
    }


def main():
    """Run comprehensive robustness testing."""

    print("=" * 70)
    print("ROBUSTNESS TESTING ACROSS MARKET CONDITIONS")
    print("=" * 70)
    print("Optimal config: 10k paths, 80 epochs, 4x128, ES(p=0.9)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Optimal configuration from previous experiments
    config = {
        "n_paths": 10000,
        "n_epochs": 80,
        "n_layers": 4,
        "n_units": 128,
        "loss": "expected_shortfall",
        "loss_param": 0.9,
        "train_seed": 42,
        "test_seed": 888,
    }

    # Base scenario
    base_spot = 50000
    base_strike = 50000
    base_maturity = 14
    base_vol = 0.8
    base_drift = 0.0
    base_cost = 0.0005

    scenarios = []

    # 1. Varying strikes (moneyness)
    print("\n" + "=" * 70)
    print("TEST 1: VARYING STRIKE PRICES (MONEYNESS)")
    print("=" * 70)

    for name, strike_ratio in [
        ("Deep ITM", 0.85),  # Strike at 85% of spot
        ("ATM (baseline)", 1.0),
        ("OTM", 1.15),  # Strike at 115% of spot
        ("Deep OTM", 1.3),
    ]:
        scenarios.append(
            {
                "name": name,
                "category": "Strike",
                "strike": int(base_strike * strike_ratio),
                "maturity_days": base_maturity,
                "volatility": base_vol,
                "drift": base_drift,
                "cost": base_cost,
            }
        )

    # 2. Varying maturities
    print("\n" + "=" * 70)
    print("TEST 2: VARYING MATURITIES")
    print("=" * 70)

    for name, days in [
        ("1 week", 7),
        ("2 weeks (baseline)", 14),
        ("3 weeks", 21),
        ("1 month", 30),
    ]:
        scenarios.append(
            {
                "name": name,
                "category": "Maturity",
                "strike": base_strike,
                "maturity_days": days,
                "volatility": base_vol,
                "drift": base_drift,
                "cost": base_cost,
            }
        )

    # 3. Varying volatilities
    print("\n" + "=" * 70)
    print("TEST 3: VARYING VOLATILITIES")
    print("=" * 70)

    for name, vol in [
        ("Low vol (40%)", 0.4),
        ("Medium vol (80%, baseline)", 0.8),
        ("High vol (120%)", 1.2),
        ("Very high vol (160%)", 1.6),
    ]:
        scenarios.append(
            {
                "name": name,
                "category": "Volatility",
                "strike": base_strike,
                "maturity_days": base_maturity,
                "volatility": vol,
                "drift": base_drift,
                "cost": base_cost,
            }
        )

    # Run experiments
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Testing: {scenario['name']}")
        print(
            f"  Strike: ${scenario['strike']}, Maturity: {scenario['maturity_days']}d, Vol: {scenario['volatility']:.0%}"
        )

        try:
            result = run_experiment(config, scenario)
            results.append(result)

            status = "✅ WIN" if result["beats_bs"] else "❌ LOSS"
            print(
                f"  Result: Deep ${result['deep']['mean']:.0f} vs BS ${result['bs']['mean']:.0f}"
            )
            print(
                f"  Improvement: ${result['improvement']:+.0f} ({result['rel_improvement']:+.1f}%) {status}"
            )

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback

            traceback.print_exc()

    # Summary by category
    print("\n\n" + "=" * 70)
    print("ROBUSTNESS TESTING SUMMARY")
    print("=" * 70)

    for category in ["Strike", "Maturity", "Volatility"]:
        cat_results = [r for r in results if r["scenario"]["category"] == category]
        if not cat_results:
            continue

        print(f"\n{category.upper()}:")
        print(f"{'Scenario':<30} {'Improv':>10} {'Rel %':>8} {'Status':>8}")
        print("-" * 70)

        for r in cat_results:
            name = r["scenario"]["name"][:28]
            improvement = r["improvement"]
            rel_improvement = r["rel_improvement"]
            status = "✅" if r["beats_bs"] else "❌"

            print(
                f"{name:<30} ${improvement:>9.0f} {rel_improvement:>7.1f}% {status:>8}"
            )

    # Overall analysis
    print("\n\n" + "=" * 70)
    print("OVERALL ROBUSTNESS ANALYSIS")
    print("=" * 70)

    wins = [r for r in results if r["beats_bs"]]
    win_rate = len(wins) / len(results) * 100 if results else 0

    print(f"\nWin Rate: {len(wins)}/{len(results)} ({win_rate:.1f}%)")

    if wins:
        avg_improvement = sum(r["improvement"] for r in wins) / len(wins)
        print(f"Average improvement (wins): ${avg_improvement:.2f}")

        best = max(wins, key=lambda r: r["improvement"])
        print(f"\nBest scenario: {best['scenario']['name']}")
        print(
            f"  Improvement: ${best['improvement']:.2f} ({best['rel_improvement']:.1f}%)"
        )

    losses = [r for r in results if not r["beats_bs"]]
    if losses:
        avg_loss = sum(abs(r["improvement"]) for r in losses) / len(losses)
        print(f"\nAverage loss (losses): ${avg_loss:.2f}")

        worst = min(results, key=lambda r: r["improvement"])
        print(f"\nWorst scenario: {worst['scenario']['name']}")
        print(f"  Loss: ${worst['improvement']:.2f} ({worst['rel_improvement']:.1f}%)")

    # Robustness score: percentage of scenarios where deep hedging wins
    print(f"\n🎯 Robustness Score: {win_rate:.1f}%")
    if win_rate >= 80:
        print("✅ EXCELLENT: Deep hedging is highly robust")
    elif win_rate >= 60:
        print("✅ GOOD: Deep hedging is generally robust")
    elif win_rate >= 40:
        print("⚠️  MODERATE: Deep hedging works in some conditions")
    else:
        print("❌ POOR: Deep hedging is not robust")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save
    import json

    with open("crypto/utils/robustness_testing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = main()
