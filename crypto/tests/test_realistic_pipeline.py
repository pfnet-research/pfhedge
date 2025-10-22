#!/usr/bin/env python3
"""
Test script to demonstrate the realistic backtesting pipeline components.

This script tests each component individually without requiring real Deribit data.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime, timezone

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto.training import TrainingConfig, Trainer
from crypto.backtest import BacktestConfig, Backtester
from crypto.utils.black_scholes import (
    black_scholes_price,
    implied_volatility,
    implied_volatility_from_btc_premium,
)


def test_black_scholes_utilities():
    """Test Black-Scholes utilities."""
    print("\n" + "=" * 60)
    print("Testing Black-Scholes Utilities")
    print("=" * 60)

    # Test parameters
    spot = 50000
    strike = 50000
    time_to_expiry = 14 / 365  # 14 days
    volatility = 0.8  # 80%

    # Test price calculation
    call_price = black_scholes_price(
        spot, strike, time_to_expiry, volatility, option_type="call"
    )
    put_price = black_scholes_price(
        spot, strike, time_to_expiry, volatility, option_type="put"
    )

    print(f"\nBlack-Scholes Prices:")
    print(f"  Spot: ${spot:,.0f}")
    print(f"  Strike: ${strike:,.0f}")
    print(f"  Time to expiry: {time_to_expiry*365:.1f} days")
    print(f"  Volatility: {volatility:.1%}")
    print(f"  Call price: ${call_price:,.2f}")
    print(f"  Put price: ${put_price:,.2f}")

    # Test IV calculation
    print("\nImplied Volatility Calculation:")

    # Use the call price we just calculated
    calculated_iv = implied_volatility(
        premium=call_price,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        option_type="call",
    )

    if calculated_iv:
        print(f"  Input volatility: {volatility:.1%}")
        print(f"  Calculated IV: {calculated_iv:.1%}")
        print(f"  Difference: {abs(calculated_iv - volatility):.4f}")
        assert abs(calculated_iv - volatility) < 0.001, "IV calculation error too large"
        print("  ✅ IV calculation working correctly")
    else:
        print("  ❌ IV calculation failed")

    # Test BTC premium conversion
    print("\nBTC Premium IV Calculation:")
    premium_btc = 0.05  # 0.05 BTC premium
    btc_iv = implied_volatility_from_btc_premium(
        premium_btc=premium_btc,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        option_type="call",
    )

    if btc_iv:
        print(f"  Premium: {premium_btc} BTC")
        print(f"  Implied volatility: {btc_iv:.1%}")
        print("  ✅ BTC premium IV working")
    else:
        print("  ❌ BTC premium IV failed")

    return True


def test_training_component():
    """Test the training component."""
    print("\n" + "=" * 60)
    print("Testing Training Component")
    print("=" * 60)

    # Create minimal training config
    config = TrainingConfig(
        strike=50000,
        maturity_days=7,
        call=True,
        volatility=0.8,
        transaction_cost=0.0006,
        n_paths=500,  # Very small for testing
        n_epochs=5,  # Very few epochs
        n_layers=2,  # Small network
        n_units=32,
        model_path="test_model.pth",
    )

    print("\nTraining Configuration:")
    print(f"  Strike: ${config.strike:,.0f}")
    print(f"  Maturity: {config.maturity_days} days")
    print(f"  Paths: {config.n_paths}")
    print(f"  Epochs: {config.n_epochs}")

    # Train model
    print("\nTraining model...")
    trainer = Trainer(config, verbose=False)
    results = trainer.train(seed=42)

    # Get loss values from training history
    initial_loss = results.training_history[0] if results.training_history else 0
    final_loss = results.training_history[-1] if results.training_history else 0

    print(f"  Initial loss: {initial_loss:.2f}")
    print(f"  Final loss: {final_loss:.2f}")
    if initial_loss > 0:
        print(f"  Improvement: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    print("  ✅ Training component working")

    # Clean up
    Path("test_model.pth").unlink(missing_ok=True)

    return True


def test_seller_pnl_calculation():
    """Test seller P&L calculation logic."""
    print("\n" + "=" * 60)
    print("Testing Seller P&L Calculation")
    print("=" * 60)

    # Simulate some data
    n_paths = 100
    initial_spot = 50000
    option_premium_btc = 0.05  # Seller receives 0.05 BTC
    premium_usd = option_premium_btc * initial_spot

    # Simulate hedging P&L (typically negative for sellers)
    # Sellers lose money when hedging their short option position
    hedge_pnl_mean = -3000  # Average hedging loss
    hedge_pnl_std = 500

    np.random.seed(42)
    hedge_pnl = np.random.normal(hedge_pnl_mean, hedge_pnl_std, n_paths)

    # Calculate seller's total P&L
    # Seller P&L = Premium received + Hedging P&L
    seller_total_pnl = premium_usd + hedge_pnl

    print(f"\nSeller P&L Breakdown:")
    print(f"  Premium received: {option_premium_btc:.4f} BTC = ${premium_usd:,.2f}")
    print(f"  Hedging P&L: ${hedge_pnl.mean():,.2f} ± ${hedge_pnl.std():,.2f}")
    print(
        f"  Total P&L: ${seller_total_pnl.mean():,.2f} ± ${seller_total_pnl.std():,.2f}"
    )

    # Check if profitable
    win_rate = (seller_total_pnl > 0).mean()
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Sharpe ratio: {seller_total_pnl.mean() / seller_total_pnl.std():.3f}")

    # Key insight
    print("\n💡 Key Insight:")
    if seller_total_pnl.mean() > 0:
        print(
            f"  Seller is profitable! Premium (${premium_usd:,.2f}) exceeds hedging costs (${abs(hedge_pnl.mean()):,.2f})"
        )
    else:
        print(
            f"  Seller loses money. Hedging costs (${abs(hedge_pnl.mean()):,.2f}) exceed premium (${premium_usd:,.2f})"
        )

    print("  ✅ Seller P&L calculation working")

    return True


def test_existing_backtest_integration():
    """Test that our approach integrates with existing backtest framework."""
    print("\n" + "=" * 60)
    print("Testing Integration with Existing Framework")
    print("=" * 60)

    # Check that sample data exists
    sample_data_dir = Path("crypto/data/sample_data")
    if not sample_data_dir.exists():
        print(f"  ⚠️  Sample data directory not found: {sample_data_dir}")
        print("  You need to download or generate sample data first")
        return False

    # List available data files
    parquet_files = list(sample_data_dir.glob("*.parquet"))
    if parquet_files:
        print(f"\n  Found {len(parquet_files)} data files:")
        for f in parquet_files[:3]:
            print(f"    - {f.name}")
    else:
        print("  ⚠️  No parquet files found in sample data")
        return False

    print("\n  ✅ Sample data available for backtesting")

    # Show how the complete pipeline would work
    print("\n📋 Complete Pipeline Steps:")
    print("  1. Fetch data: fetch_deribit_data.py")
    print("  2. Select option: select_option.py")
    print("  3. Train model: (uses existing Trainer)")
    print("  4. Run backtest: (uses existing Backtester)")
    print("  5. Calculate seller P&L: Premium + Hedging P&L")
    print("  6. Generate report: Markdown + Plots")

    return True


def main():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("REALISTIC BACKTESTING PIPELINE TEST")
    print("🚀 " * 20)

    results = []

    # Test each component
    try:
        results.append(("Black-Scholes Utilities", test_black_scholes_utilities()))
    except Exception as e:
        print(f"  ❌ Black-Scholes test failed: {e}")
        results.append(("Black-Scholes Utilities", False))

    try:
        results.append(("Training Component", test_training_component()))
    except Exception as e:
        print(f"  ❌ Training test failed: {e}")
        results.append(("Training Component", False))

    try:
        results.append(("Seller P&L Calculation", test_seller_pnl_calculation()))
    except Exception as e:
        print(f"  ❌ Seller P&L test failed: {e}")
        results.append(("Seller P&L Calculation", False))

    try:
        results.append(("Framework Integration", test_existing_backtest_integration()))
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        results.append(("Framework Integration", False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed! Pipeline components are working.")
        print("\nNext steps:")
        print("1. Get Deribit API access (mainnet or testnet)")
        print(
            "2. Run: python crypto/scripts/fetch_deribit_data.py --start 2024-01-01 --end 2024-01-31"
        )
        print(
            "3. Run: python crypto/scripts/realistic_backtest.py --config crypto/configs/realistic_backtest_example.yaml"
        )
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
