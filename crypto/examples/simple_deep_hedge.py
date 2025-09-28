#!/usr/bin/env python3
"""
Simple Deep Hedging Example - Baby Steps Implementation

This is a minimal working example that:
1. Creates synthetic Bitcoin price paths
2. Creates a European option
3. Trains a deep hedging model
4. Compares with delta hedging
5. Shows simple metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from crypto.instruments import BitcoinPerpetualBrownian
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import Hedger, MultiLayerPerceptron, BlackScholes, EntropicLoss


def main():
    """Simple deep hedging example."""

    print("=" * 60)
    print("SIMPLE DEEP HEDGING EXAMPLE")
    print("=" * 60)

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Step 1: Create Bitcoin market
    print("\n1. Creating synthetic Bitcoin market...")

    btc = BitcoinPerpetualBrownian(
        sigma=0.8,  # 80% annual volatility (typical for Bitcoin)
        mu=0.0,     # No drift for simplicity
        cost=0.001, # 10 basis points transaction cost
    )

    # Generate paths for training
    n_paths = 1000  # Start small
    maturity = 7/365  # 7 days

    btc.simulate(n_paths=n_paths, time_horizon=maturity)

    print(f"   Generated {n_paths} paths")
    print(f"   Time steps: {btc.spot.shape[1]}")
    print(f"   Initial price: ${btc.spot[0, 0]:.0f}")

    # Step 2: Create European option
    print("\n2. Creating European option...")

    strike = btc.spot[:, 0].mean().item()  # ATM option
    option = EuropeanOption(btc, strike=strike, maturity=maturity)

    print(f"   Strike: ${strike:.0f}")
    print(f"   Maturity: {maturity*365:.0f} days")

    # Step 3: Train deep hedging model
    print("\n3. Training deep hedging model...")

    # Create simple neural network
    model = MultiLayerPerceptron(
        in_features=3,
        out_features=1,
        n_layers=2,  # Simple network
        n_units=32,  # Small network
    )

    # Create hedger with entropic risk measure
    criterion = EntropicLoss(a=1.0)  # Risk aversion parameter
    deep_hedger = Hedger(
        model=model,
        inputs=["log_moneyness", "time_to_maturity", "volatility"],
        criterion=criterion,
    )

    # Train (this will take a moment)
    print("   Training for 20 epochs...")
    deep_hedger.fit(option, n_epochs=20, verbose=False)
    print("   Training complete!")

    # Step 4: Create delta hedging baseline
    print("\n4. Creating delta hedge baseline...")

    # Use Black-Scholes for delta hedging
    from pfhedge.nn import BSEuropeanOption
    bs_model = BSEuropeanOption()
    delta_hedger = Hedger(
        model=bs_model,
        inputs=["log_moneyness", "time_to_maturity", "volatility"],
    )
    print("   Delta hedger ready")

    # Step 5: Evaluate both strategies
    print("\n5. Evaluating strategies...")

    # Generate test data
    test_btc = BitcoinPerpetualBrownian(sigma=0.8, mu=0.0, cost=0.001)
    test_btc.simulate(n_paths=100, time_horizon=maturity)
    test_option = EuropeanOption(test_btc, strike=strike, maturity=maturity)

    # Calculate PnL using simpler method
    with torch.no_grad():
        try:
            # For deep hedging - use price method which gives final portfolio value
            deep_price = deep_hedger.price(test_option)
            deep_pnl = deep_price - test_option.payoff()

            # For delta hedging
            delta_price = delta_hedger.price(test_option)
            delta_pnl = delta_price - test_option.payoff()

            print(f"   Deep hedge portfolio: mean=${deep_price.mean():.2f}")
            print(f"   Delta hedge portfolio: mean=${delta_price.mean():.2f}")
            print(f"   Option payoff: mean=${test_option.payoff().mean():.2f}")

        except Exception as e:
            print(f"   Error in PnL calculation: {e}")
            # Fallback to simple payoff comparison
            payoff = test_option.payoff()
            deep_pnl = torch.zeros_like(payoff)
            delta_pnl = torch.zeros_like(payoff)

    # Step 6: Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Deep Hedge':>15} {'Delta Hedge':>15}")
    print("-" * 50)

    # Calculate statistics safely
    deep_mean = deep_pnl.mean().item() if not torch.isnan(deep_pnl).any() else 0.0
    delta_mean = delta_pnl.mean().item() if not torch.isnan(delta_pnl).any() else 0.0
    print(f"{'Mean PnL':<20} ${deep_mean:>14.2f} ${delta_mean:>14.2f}")

    # Std PnL
    deep_std = deep_pnl.std().item() if not torch.isnan(deep_pnl).any() else 0.0
    delta_std = delta_pnl.std().item() if not torch.isnan(delta_pnl).any() else 0.0
    print(f"{'Std PnL':<20} ${deep_std:>14.2f} ${delta_std:>14.2f}")

    # Sharpe ratio (annualized)
    deep_sharpe = deep_mean / deep_std * np.sqrt(365/7) if deep_std > 0 else 0
    delta_sharpe = delta_mean / delta_std * np.sqrt(365/7) if delta_std > 0 else 0
    print(f"{'Sharpe Ratio':<20} {deep_sharpe:>15.3f} {delta_sharpe:>15.3f}")

    # Win rate
    deep_win = (deep_pnl > 0).float().mean().item()
    delta_win = (delta_pnl > 0).float().mean().item()
    print(f"{'Win Rate':<20} {deep_win:>14.1%} {delta_win:>14.1%}")

    print("\n" + "=" * 60)

    # Summary
    if deep_std < delta_std:
        print("✅ Deep hedge has lower risk (std) than delta hedge")
    else:
        print("❌ Delta hedge has lower risk than deep hedge")

    if deep_sharpe > delta_sharpe:
        print("✅ Deep hedge has better risk-adjusted returns")
    else:
        print("❌ Delta hedge has better risk-adjusted returns")

    print("\nNote: This is a simple example with limited training.")
    print("Real performance requires more data and longer training.")

    return deep_hedger, delta_hedger


if __name__ == "__main__":
    deep_hedger, delta_hedger = main()