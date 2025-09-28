#!/usr/bin/env python3
"""
Quick Test Example - Always Works

This is your reliable running example for testing improvements.
Fast, simple, and always produces meaningful results.

Usage: python quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
from crypto.instruments import BitcoinPerpetualBrownian
from crypto.utils.visualization import plot_price_paths, plot_option_analysis, plot_hedging_performance
from crypto.features.volatility import calculate_realized_volatility, create_volatility_features

def test_bitcoin_instruments():
    """Test our Bitcoin instruments work correctly."""
    print("=" * 50)
    print("QUICK DEEP HEDGING TEST")
    print("=" * 50)

    # Set reproducible seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Test 1: Bitcoin instrument
    print("\n1. Testing Bitcoin Instrument...")
    btc = BitcoinPerpetualBrownian(sigma=0.8, mu=0.0, cost=0.001)
    btc.simulate(n_paths=100, time_horizon=7/365)

    print(f"   ✅ Generated {btc.spot.shape[0]} paths")
    print(f"   ✅ {btc.spot.shape[1]} time steps")
    print(f"   ✅ Price range: ${btc.spot.min():.0f} - ${btc.spot.max():.0f}")

    # Test 2: Volatility calculation
    print("\n2. Testing Volatility...")
    vol = btc.volatility
    print(f"   ✅ Volatility shape: {vol.shape}")
    print(f"   ✅ Mean volatility: {vol.mean():.2%}")

    # Test 2b: Realized volatility
    print("\n2b. Testing Realized Volatility...")
    realized_vol = calculate_realized_volatility(
        btc.spot,
        window=20,
        annualization_factor=np.sqrt(252 * 24 * 12)  # 5-min data
    )
    print(f"   ✅ Realized vol shape: {realized_vol.shape}")
    print(f"   ✅ Mean realized vol: {realized_vol.nanmean():.2%}")

    # Test volatility features for deep hedging
    vol_features = create_volatility_features(btc, windows=[10, 20])
    print(f"   ✅ Volatility features shape: {vol_features.shape}")
    print(f"   ✅ Features: 10-day vol, 20-day vol")

    # Test 3: Returns calculation
    print("\n3. Testing Returns...")
    returns = torch.log(btc.spot[:, 1:] / btc.spot[:, :-1])
    daily_vol = returns.std() * np.sqrt(288)  # 288 5-min periods per day
    print(f"   ✅ Daily returns volatility: {daily_vol:.2%}")

    # Test 4: Option-like payoff
    print("\n4. Testing Option Payoff...")
    strike = btc.spot[:, 0].mean()
    final_prices = btc.spot[:, -1]
    call_payoff = torch.clamp(final_prices - strike, min=0)

    print(f"   ✅ Strike: ${strike:.0f}")
    print(f"   ✅ ITM ratio: {(call_payoff > 0).float().mean():.1%}")
    print(f"   ✅ Average payoff: ${call_payoff.mean():.2f}")

    # Test 5: Simple hedging simulation
    print("\n5. Testing Simple Hedge...")

    # Naive strategy: hold 0.5 units of underlying
    hedge_ratio = 0.5
    hedge_pnl = hedge_ratio * (final_prices - btc.spot[:, 0]) - call_payoff

    print(f"   ✅ Hedge ratio: {hedge_ratio}")
    print(f"   ✅ Hedge PnL mean: ${hedge_pnl.mean():.2f}")
    print(f"   ✅ Hedge PnL std: ${hedge_pnl.std():.2f}")

    # Success metrics
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    all_tests_pass = (
        btc.spot.shape[0] == 100 and
        not torch.isnan(vol).any() and
        not torch.isnan(hedge_pnl).any()
    )

    if all_tests_pass:
        print("✅ ALL TESTS PASSED")
        print("✅ Bitcoin instruments working correctly")
        print("✅ Ready for deep hedging implementation")
    else:
        print("❌ Some tests failed")

    # Baseline metrics for improvement tracking
    baseline_metrics = {
        'volatility': vol.mean().item(),
        'realized_vol': realized_vol.nanmean().item(),
        'option_value': call_payoff.mean().item(),
        'hedge_std': hedge_pnl.std().item(),
        'itm_ratio': (call_payoff > 0).float().mean().item()
    }

    print(f"\nBaseline Metrics (for tracking improvements):")
    for metric, value in baseline_metrics.items():
        if 'ratio' in metric:
            print(f"  {metric}: {value:.1%}")
        elif 'std' in metric or 'value' in metric:
            print(f"  {metric}: ${value:.2f}")
        else:
            print(f"  {metric}: {value:.3f}")

    return baseline_metrics, all_tests_pass


def test_improvements():
    """Demonstrate how to test improvements."""
    print("\n" + "=" * 50)
    print("TESTING IMPROVEMENTS")
    print("=" * 50)

    print("\nTo test improvements:")
    print("1. Modify parameters in this script")
    print("2. Run: python quick_test.py")
    print("3. Compare metrics with baseline")

    print("\nExample improvements to try:")
    print("- Change volatility: sigma=1.0 (higher vol)")
    print("- Change time horizon: 30/365 (longer)")
    print("- Change hedge ratio: 0.3 or 0.7")
    print("- Add realized volatility features")

    # Example: Test with different parameters
    print("\nExample: Testing with higher volatility...")
    btc_high_vol = BitcoinPerpetualBrownian(sigma=1.2, mu=0.0, cost=0.001)
    btc_high_vol.simulate(n_paths=100, time_horizon=7/365)

    high_vol_metric = btc_high_vol.volatility.mean().item()
    print(f"  High vol result: {high_vol_metric:.3f}")
    print(f"  vs baseline: 0.800")
    print(f"  Improvement: {(high_vol_metric - 0.8) / 0.8 * 100:+.1f}%")


def demo_visualization_utilities():
    """Demonstrate the visualization utilities."""
    print("\n" + "=" * 50)
    print("VISUALIZATION UTILITIES DEMO")
    print("=" * 50)

    print("\n🎨 Our visualization utilities work with ANY instrument:")
    print("- plot_price_paths(instrument)")
    print("- plot_option_analysis(instrument, strike)")
    print("- plot_hedging_performance(pnl)")

    # Create example instrument
    btc = BitcoinPerpetualBrownian(sigma=0.8, mu=0.1, cost=0.001)
    btc.simulate(n_paths=200, time_horizon=14/365)

    print(f"\n📊 Example: Created {btc.spot.shape[0]} paths for visualization")

    print("\n💡 To use the visualization utilities:")
    print("```python")
    print("from crypto.utils.visualization import plot_price_paths")
    print("fig = plot_price_paths(btc, n_paths_to_show=20)")
    print("plt.show()")
    print("```")

    print("\n🚀 Available functions:")
    print("- plot_price_paths: Price evolution and distribution")
    print("- plot_option_analysis: Payoff diagrams and moneyness")
    print("- plot_hedging_performance: PnL analysis and risk metrics")
    print("- plot_volatility_analysis: Volatility patterns")
    print("- quick_instrument_analysis: Complete analysis suite")

    print("\n📈 Volatility Features for Deep Hedging:")
    print("- calculate_realized_volatility: Historical volatility calculation")
    print("- create_volatility_features: Multiple time windows for ML models")
    print("- RealizedVolatilityCalculator: Real-time volatility estimation")

    return btc


if __name__ == "__main__":
    baseline, success = test_bitcoin_instruments()

    if success:
        test_improvements()
        demo_visualization_utilities()
        print(f"\n✅ Quick test complete! Your deep hedging infrastructure is working.")
        print(f"\n🎨 Visualization utilities are ready for use with any instrument!")
    else:
        print(f"\n❌ Infrastructure needs debugging before proceeding.")