#!/usr/bin/env python3
"""
Aggressive optimization - push training harder.

Tests very long training, massive capacity, and many paths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from datetime import datetime
from crypto.instruments import create_bitcoin_option_from_config
from crypto.strategies import create_deep_hedger, compare_hedge_performance, calculate_bs_hedge_pnl


def run_experiment(config):
    """Run a single optimization experiment."""

    # Base parameters
    base_config = {
        'strike': 50000,
        'maturity_days': 14,
        'volatility': 0.8,
        'drift': 0.0,
        'cost': 0.0005,
        'train_seed': 42,
        'test_seed': 888,
    }

    # Training
    train_config = {
        **base_config,
        'call': True,
        'cost': 0.0,
        'sigma': base_config['volatility'],
        'mu': base_config['drift'],
        'underlier_cost': base_config['cost'],
        'n_paths': config['n_paths'],
        'seed': base_config['train_seed']
    }

    option_train, _ = create_bitcoin_option_from_config(train_config)

    # Create and train hedger
    hedger = create_deep_hedger(
        n_layers=config['n_layers'],
        n_units=config['n_units'],
        risk_measure=config['loss'],
        risk_param=config['loss_param'],
    )

    history = hedger.fit(
        option_train,
        n_paths=config['n_paths'],
        n_epochs=config['n_epochs'],
        verbose=False
    )

    # Test
    test_config = {
        **base_config,
        'call': True,
        'cost': 0.0,
        'sigma': base_config['volatility'],
        'mu': base_config['drift'],
        'underlier_cost': base_config['cost'],
        'n_paths': 200,
        'seed': base_config['test_seed']
    }

    option_test, _ = create_bitcoin_option_from_config(test_config)

    with torch.no_grad():
        deep_pnl = hedger.compute_cum_pl(option_test).squeeze()
        bs_delta = option_test.black_scholes_delta()
        payoffs = option_test.payoff()
        spots = option_test.underlier.spot
        bs_pnl = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, base_config['cost'])

    results = compare_hedge_performance(deep_pnl, bs_pnl)

    improvement = results['Deep Hedge']['mean'] - results['Black-Scholes']['mean']
    beats_bs = improvement > 0

    return {
        'name': config['name'],
        'config': config,
        'deep': results['Deep Hedge'],
        'bs': results['Black-Scholes'],
        'improvement': improvement,
        'beats_bs': beats_bs,
        'training_improvement': (history[0] - history[-1]) / history[0] * 100 if len(history) > 0 else 0
    }


def main():
    """Run aggressive optimization experiments."""

    print("="*70)
    print("AGGRESSIVE OPTIMIZATION - PUSH THE LIMITS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    experiments = [
        # Very long training
        {'name': 'Ultra long (150 epochs)', 'n_epochs': 150, 'n_paths': 3000, 'n_layers': 4, 'n_units': 128, 'loss': 'expected_shortfall', 'loss_param': 0.5},
        {'name': 'Marathon (200 epochs)', 'n_epochs': 200, 'n_paths': 3000, 'n_layers': 4, 'n_units': 128, 'loss': 'expected_shortfall', 'loss_param': 0.5},

        # Massive capacity
        {'name': 'Very deep (6 layers)', 'n_epochs': 80, 'n_paths': 3000, 'n_layers': 6, 'n_units': 128, 'loss': 'expected_shortfall', 'loss_param': 0.5},
        {'name': 'Very wide (256 units)', 'n_epochs': 80, 'n_paths': 3000, 'n_layers': 4, 'n_units': 256, 'loss': 'expected_shortfall', 'loss_param': 0.5},
        {'name': 'Massive (6x256)', 'n_epochs': 100, 'n_paths': 4000, 'n_layers': 6, 'n_units': 256, 'loss': 'expected_shortfall', 'loss_param': 0.5},

        # Many training paths
        {'name': 'Many paths (5000)', 'n_epochs': 80, 'n_paths': 5000, 'n_layers': 4, 'n_units': 128, 'loss': 'expected_shortfall', 'loss_param': 0.5},
        {'name': 'Max paths (10000)', 'n_epochs': 80, 'n_paths': 10000, 'n_layers': 4, 'n_units': 128, 'loss': 'expected_shortfall', 'loss_param': 0.5},
    ]

    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Testing: {exp['name']}")
        print(f"  Config: {exp['n_epochs']}ep, {exp['n_paths']}paths, {exp['n_layers']}x{exp['n_units']}, {exp['loss']}({exp['loss_param']})")

        try:
            result = run_experiment(exp)
            results.append(result)

            status = "✅ WIN" if result['beats_bs'] else "❌ LOSS"
            print(f"  Result: Deep ${result['deep']['mean']:.0f} vs BS ${result['bs']['mean']:.0f} ({result['improvement']:+.0f}) {status}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

    # Summary
    print("\n\n" + "="*70)
    print("AGGRESSIVE OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\n{'Configuration':<30} {'Deep PnL':>12} {'Improv':>10} {'Status':>8}")
    print("-"*70)

    for r in results:
        name = r['name'][:28]
        deep_pnl = r['deep']['mean']
        improvement = r['improvement']
        status = "✅" if r['beats_bs'] else "❌"

        print(f"{name:<30} ${deep_pnl:>10.0f} ${improvement:>9.0f} {status:>8}")

    # Find best
    if results:
        wins = [r for r in results if r['beats_bs']]
        if wins:
            best = max(wins, key=lambda r: r['improvement'])
            print(f"\n{'='*70}")
            print(f"🏆 BEST CONFIGURATION: {best['name']}")
            print(f"{'='*70}")
            print(f"Deep Hedge PnL: ${best['deep']['mean']:,.2f}")
            print(f"BS PnL: ${best['bs']['mean']:,.2f}")
            print(f"Improvement: ${best['improvement']:,.2f}")
            print(f"Deep Sharpe: {best['deep']['sharpe']:.3f}")
            print(f"Training improvement: {best['training_improvement']:.1f}%")
            print(f"\nConfiguration:")
            print(f"  Epochs: {best['config']['n_epochs']}")
            print(f"  Paths: {best['config']['n_paths']}")
            print(f"  Architecture: {best['config']['n_layers']} layers x {best['config']['n_units']} units")
            print(f"  Loss: {best['config']['loss']}({best['config']['loss_param']})")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save
    import json
    with open('crypto/utils/aggressive_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = main()
