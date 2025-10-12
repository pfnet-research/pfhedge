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
    n_paths = 10000
    n_epochs = 80
    test_n_paths = 200

    # Bitcoin option parameters
    strike = 50000
    maturity_days = 14
    volatility = 0.8
    drift = 0.0
    cost = 0.0005  # 0.05% transaction cost (Deribit taker fee)
    dt = 8 / 24 / 365  # 8-hour time steps (matches funding interval)

    print("="*60)
    print("BITCOIN DEEP HEDGING - Training and Evaluation")
    print("="*60)
    print(f"Configuration:")
    print(f"  Strike: ${strike}")
    print(f"  Maturity: {maturity_days} days")
    print(f"  Volatility: {volatility:.1%}")
    print(f"  Transaction cost: {cost:.2%}")
    print(f"  Time step (dt): {dt*365*24:.1f} hours")
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
        'dt': dt,
        'n_paths': n_paths,
        'seed': train_seed
    }
    option_train, _ = create_bitcoin_option_from_config(train_config)

    # Create and train hedger using utility function (optimal: 4 layers x 128 units, ES p=0.9)
    deep_hedger = create_deep_hedger(n_layers=4, n_units=128, risk_param=0.9)

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
        'dt': dt,
        'n_paths': test_n_paths,
        'seed': test_seed
    }
    option_test, _ = create_bitcoin_option_from_config(test_config)

    print(f"\nGenerating hedging strategies...")

    with torch.no_grad():
        # Compute deep hedging strategy
        deep_hedge_positions = deep_hedger.compute_hedge(option_test).squeeze()

        # Get spot prices and funding
        spots = option_test.underlier.spot
        # Funding tensors
        funding_rate = option_test.underlier.funding_rate  # (n_paths, n_steps)
        funding_times = option_test.underlier.funding_payment_times()  # (n_steps,)

        # Deep hedger PnL including funding
        deep_hedge_pnl = deep_hedger.compute_cum_pl(option_test).squeeze()
        # Subtract funding for the deep hedger positions
        from crypto.strategies.deep_hedge_utils import compute_funding_cum_cost
        deep_funding = compute_funding_cum_cost(
            spots=spots,
            positions=deep_hedge_positions,
            funding_rate=funding_rate,
            funding_times=funding_times,
        )
        deep_hedge_pnl = deep_hedge_pnl - deep_funding

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
    # Include funding in BS baseline as well
    bs_hedge_pnl = calculate_bs_hedge_pnl(
        spots, bs_delta, payoffs, cost,
        funding_rate=funding_rate,
        funding_times=funding_times,
    )

    print(f"\n✅ BS delta shape: {bs_delta.shape}")
    print(f"✅ BS hedge PnL shape: {bs_hedge_pnl.shape}")

    # ========== Option Premium and Total PnL ==========

    print("\n" + "="*60)
    print("TOTAL PNL CALCULATION (Premium + Hedging)")
    print("="*60)

    # Calculate option premium (Black-Scholes price at t=0)
    from pfhedge.nn.functional import bs_european_price
    initial_spot = option_test.underlier.spot[:, 0]
    option_premium = bs_european_price(
        log_moneyness=torch.log(initial_spot / strike),
        time_to_maturity=torch.tensor(maturity_days / 365),
        volatility=torch.tensor(volatility),
        strike=strike,
        call=True
    ).mean().item()
    print(f"\nOption premium (BS price at t=0): ${option_premium:.2f}")

    # Total PnL = Premium received - Payoff paid + Hedging gains/losses
    # Note: Hedging PnL already includes -payoff at maturity
    # So: Total PnL = Premium + Hedging PnL
    deep_total_pnl = option_premium + deep_hedge_pnl[:, -1]
    bs_total_pnl = option_premium + bs_hedge_pnl[:, -1]

    print(f"\nDeep hedge total PnL: ${deep_total_pnl.mean().item():.2f} ± ${deep_total_pnl.std().item():.2f}")
    print(f"BS total PnL: ${bs_total_pnl.mean().item():.2f} ± ${bs_total_pnl.std().item():.2f}")
    print(f"Difference: ${(deep_total_pnl.mean() - bs_total_pnl.mean()).item():.2f}")

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
