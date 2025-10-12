#!/usr/bin/env python3
"""
Bitcoin Deep Hedging - Following snowball_hedge.py pattern

This script demonstrates deep hedging for Bitcoin European options,
following the same structure as examples/snowball_hedge.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import matplotlib.pyplot as plt

from crypto.instruments import create_bitcoin_option_from_config
from crypto.strategies import (
    create_deep_hedger,
    calculate_bs_hedge_pnl,
    compare_hedge_performance,
    print_performance_comparison,
)
from crypto.utils import plot_hedge_comparison


def main():
    """Main function following snowball_hedge.py pattern."""

    # ========== Parameters ==========
    train_seed = 42
    test_seed = 888
    n_paths = 1000
    n_epochs = 20
    test_n_paths = 10

    # Bitcoin option parameters
    strike = 50000
    maturity_days = 14
    maturity = maturity_days / 365
    volatility = 0.8
    drift = 0.0
    cost = 0.001  # 0.1% transaction cost

    print("="*60)
    print("BITCOIN DEEP HEDGING - Training and Evaluation")
    print("="*60)
    print(f"Configuration:")
    print(f"  Strike: ${strike}")
    print(f"  Maturity: {maturity_days} days")
    print(f"  Volatility: {volatility:.1%}")
    print(f"  Transaction cost: {cost:.2%}")
    print(f"  Training paths: {n_paths}")
    print(f"  Training epochs: {n_epochs}")
    print(f"  Test paths: {test_n_paths}")

    # ========== Train Deep Hedger ==========

    print("\n" + "="*60)
    print("TRAINING DEEP HEDGER")
    print("="*60)

    # Create training option using config
    train_config = {
        'strike': strike,
        'maturity_days': maturity_days,
        'call': True,
        'cost': 0.0,  # No option transaction cost, only underlier cost
        'sigma': volatility,
        'mu': drift,
        'underlier_cost': cost,
        'n_paths': n_paths,
        'seed': train_seed
    }
    option_train, _ = create_bitcoin_option_from_config(train_config)

    # Create and train hedger using utility function
    deep_hedger = create_deep_hedger(n_layers=3, n_units=64, risk_param=0.5)

    print(f"\nTraining for {n_epochs} epochs...")
    history = deep_hedger.fit(
        option_train,
        n_paths=n_paths,
        n_epochs=n_epochs,
        verbose=True
    )

    print(f"\n✅ Training complete!")
    if len(history) > 0:
        print(f"  Initial loss: {history[0]:.6f}")
        print(f"  Final loss: {history[-1]:.6f}")
        improvement = (history[0] - history[-1]) / history[0] * 100
        print(f"  Improvement: {improvement:.1f}%")

    # ========== Test Deep Hedger ==========

    print("\n" + "="*60)
    print("TESTING DEEP HEDGER")
    print("="*60)

    # Create test option using config with fewer paths for visualization
    test_config = {
        'strike': strike,
        'maturity_days': maturity_days,
        'call': True,
        'cost': 0.0,  # No option transaction cost, only underlier cost
        'sigma': volatility,
        'mu': drift,
        'underlier_cost': cost,
        'n_paths': test_n_paths,
        'seed': test_seed
    }
    option_test, _ = create_bitcoin_option_from_config(test_config)

    print(f"\nGenerating hedging strategies...")

    with torch.no_grad():
        # Compute deep hedging strategy
        deep_hedge_positions = deep_hedger.compute_hedge(option_test).squeeze()
        deep_hedge_pnl = deep_hedger.compute_cum_pl(option_test).squeeze()

        # Get spot prices for visualization
        spots = option_test.underlier.spot

    print(f"✅ Deep hedge positions shape: {deep_hedge_positions.shape}")
    print(f"✅ Deep hedge PnL shape: {deep_hedge_pnl.shape}")
    print(f"✅ Spot prices shape: {spots.shape}")

    # ========== Black-Scholes Delta Baseline ==========

    print("\n" + "="*60)
    print("BLACK-SCHOLES DELTA BASELINE")
    print("="*60)

    # Calculate BS delta and PnL using utility function
    bs_delta = option_test.black_scholes_delta()
    payoffs = option_test.payoff()

    # Use utility function for PnL calculation
    bs_hedge_pnl = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, cost)

    print(f"\n✅ BS delta shape: {bs_delta.shape}")
    print(f"✅ BS hedge PnL shape: {bs_hedge_pnl.shape}")

    # ========== Performance Comparison ==========

    # Use utility function for comparison
    results = compare_hedge_performance(deep_hedge_pnl, bs_hedge_pnl)
    print_performance_comparison(results)

    # ========== Visualization ==========

    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Use utility function for comprehensive hedge comparison visualization
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'bitcoin_hedge_{n_epochs}epochs.png')

    fig = plot_hedge_comparison(
        deep_hedge_positions=deep_hedge_positions,
        bs_delta=bs_delta,
        deep_hedge_pnl=deep_hedge_pnl,
        bs_hedge_pnl=bs_hedge_pnl,
        spots=spots,
        strike=strike,
        training_history=history,
        performance_results=results,
        path_idx=0,
        save_path=output_file
    )
    plt.close()  # Close instead of show for non-interactive mode
    print(f"\n✅ Saved figure to {output_file}")

    print("\n" + "="*60)
    print("✅ Bitcoin deep hedging complete!")
    print("="*60)

    return {
        'deep_hedger': deep_hedger,
        'history': history,
        'deep_hedge_positions': deep_hedge_positions,
        'deep_hedge_pnl': deep_hedge_pnl,
        'bs_delta': bs_delta,
        'bs_hedge_pnl': bs_hedge_pnl,
        'spots': spots,
        'option_test': option_test
    }


if __name__ == "__main__":
    results = main()
