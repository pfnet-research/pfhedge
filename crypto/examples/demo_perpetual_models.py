"""
Demo of Bitcoin perpetual models for training vs backtesting.
"""
import sys
import os
import torch

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crypto.instruments import (
    BitcoinPerpetualBrownian,
    BitcoinPerpetualHistorical
)
from crypto.data.loader import CryptoDataLoader


def demo_training_model():
    """Demo using Brownian model for deep hedging training."""
    print("=" * 60)
    print("TRAINING MODEL - BitcoinPerpetualBrownian")
    print("=" * 60)

    # Create Brownian model for training
    btc = BitcoinPerpetualBrownian(
        sigma=0.8,      # 80% annual volatility (typical for Bitcoin)
        mu=0.1,         # 10% annual drift
        funding_mean=0.0001,  # Average funding rate
        funding_std=0.0002,   # Funding volatility
    )

    # Generate many paths for Monte Carlo training
    n_paths = 1000
    time_horizon = 30/365  # 30 days

    print(f"\nGenerating {n_paths} synthetic paths for training...")
    btc.simulate(n_paths=n_paths, time_horizon=time_horizon)

    print(f"Spot prices shape: {btc.spot.shape}")
    print(f"Number of time steps: {btc.spot.shape[1]}")
    print(f"Time step size: {btc.dt:.4f} days ({btc.dt * 24 * 60:.1f} minutes)")

    # Check path diversity (important for training)
    price_std = btc.spot[:, -1].std().item()
    price_mean = btc.spot[:, -1].mean().item()
    print(f"\nFinal prices across paths:")
    print(f"  Mean: ${price_mean:,.0f}")
    print(f"  Std:  ${price_std:,.0f}")
    print(f"  CoV:  {price_std/price_mean:.2%}")

    # Show funding statistics
    funding_mean = btc.funding_rate.mean().item()
    funding_std = btc.funding_rate.std().item()
    print(f"\nFunding rate statistics:")
    print(f"  Mean: {funding_mean:.4%}")
    print(f"  Std:  {funding_std:.4%}")

    # Example: Use with PFHedge for training
    try:
        from pfhedge.instruments import EuropeanOption
        from pfhedge.nn import Hedger, MLP

        # Create option on Bitcoin
        option = EuropeanOption(
            underlier=btc,
            strike=50000,
            maturity=30/365
        )

        print("\nOption created for deep hedging training:")
        print(f"  Strike: $50,000")
        print(f"  Maturity: 30 days")
        print(f"  Underlier: {btc}")

        # Calculate payoffs
        payoff = option.payoff()
        itm_ratio = (payoff > 0).float().mean().item()
        print(f"\nIn-the-money ratio: {itm_ratio:.1%}")

    except ImportError:
        print("\nPFHedge not available for full demo")


def demo_backtesting_model():
    """Demo using Historical model for backtesting."""
    print("\n" + "=" * 60)
    print("BACKTESTING MODEL - BitcoinPerpetualHistorical")
    print("=" * 60)

    # Load sample data
    loader = CryptoDataLoader("sample_data")

    # Load data to check if available
    try:
        perp_data = loader.load_perpetual_data()
        if perp_data is None or perp_data.empty:
            print("\nNo historical data available. Run download_historical.py first.")
            return
    except Exception as e:
        print(f"\nCouldn't load data: {e}")
        return

    # Create Historical model for backtesting
    btc = BitcoinPerpetualHistorical(
        data_loader=loader,
        cost=0.0006,  # 6 basis points transaction cost
        leverage=20.0
    )

    # Load historical data for backtesting
    time_horizon = 5/365  # 5 days

    print(f"\nLoading historical data for backtesting...")
    btc.simulate(n_paths=1, time_horizon=time_horizon)

    print(f"Spot prices shape: {btc.spot.shape}")
    print(f"Data points loaded: {btc.spot.shape[1]}")

    # Show actual historical statistics
    initial_price = btc.spot[0, 0].item()
    final_price = btc.spot[0, -1].item()
    returns = (final_price / initial_price - 1)

    print(f"\nHistorical price movement:")
    print(f"  Initial: ${initial_price:,.0f}")
    print(f"  Final:   ${final_price:,.0f}")
    print(f"  Return:  {returns:.2%}")

    # Calculate historical volatility
    vol = btc.volatility[0, -1].item()
    print(f"\nRealized volatility: {vol:.1%} annualized")

    # Show funding costs
    if hasattr(btc, 'funding_rate'):
        total_funding = btc.funding_rate[0].sum().item()
        print(f"Total funding over period: {total_funding:.4%}")

    print("\n" + "-" * 40)
    print("BOOTSTRAP SIMULATION")
    print("-" * 40)

    # Demo bootstrap for multiple backtesting scenarios
    print(f"\nGenerating 100 bootstrap paths from historical data...")
    btc.simulate_bootstrap(n_paths=100, time_horizon=1/365)  # 1 day windows

    print(f"Bootstrap paths shape: {btc.spot.shape}")

    # Show diversity of bootstrap samples
    final_prices = btc.spot[:, -1]
    print(f"\nBootstrap final prices:")
    print(f"  Mean: ${final_prices.mean().item():,.0f}")
    print(f"  Std:  ${final_prices.std().item():,.0f}")
    print(f"  Min:  ${final_prices.min().item():,.0f}")
    print(f"  Max:  ${final_prices.max().item():,.0f}")


def demo_comparison():
    """Compare training vs backtesting models."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print("""
Key Differences:

1. BitcoinPerpetualBrownian (Training):
   - Generates synthetic paths using stochastic models
   - Can create unlimited scenarios for Monte Carlo
   - Parameters calibrated from historical statistics
   - Used for training deep hedging models

2. BitcoinPerpetualHistorical (Backtesting):
   - Uses actual historical market data
   - Limited to available historical periods
   - Captures real market dynamics and anomalies
   - Used for backtesting strategies on real data

3. Use Cases:
   Training Phase:
   - Use BitcoinPerpetualBrownian with n_paths=10000+
   - Train neural network to learn optimal hedging
   - Explore wide range of market scenarios

   Validation Phase:
   - Use BitcoinPerpetualHistorical with bootstrap
   - Test learned strategies on real market conditions
   - Evaluate out-of-sample performance

   Production:
   - Deploy trained model with real-time data
   - Monitor performance vs historical benchmarks
    """)


def main():
    """Run all demos."""
    demo_training_model()
    demo_backtesting_model()
    demo_comparison()


if __name__ == "__main__":
    main()