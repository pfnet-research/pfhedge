#!/usr/bin/env python3
"""
Performance test for rolling volatility calculation.
"""

import torch
import time
from crypto.instruments.bitcoin_spot_brownian import BitcoinSpotBrownian


def test_performance():
    """Test performance with training-sized data."""
    print("=" * 80)
    print("Rolling Volatility Performance Test")
    print("=" * 80)

    # Create instrument with parameters matching training
    btc = BitcoinSpotBrownian(
        sigma=0.42,
        mu=0.0,
        cost=0.0001,
        dt=8/24/365,
        volatility_window=20
    )

    # Simulate paths matching training size
    n_paths = 5000
    time_horizon = 30/365

    print(f"\nSimulating {n_paths} paths over {time_horizon*365:.1f} days...")
    btc.simulate(n_paths=n_paths, time_horizon=time_horizon)

    print(f"Spot shape: {btc.spot.shape}")

    # Time the volatility calculation
    print(f"\nCalculating rolling volatility (window=20)...")
    start = time.time()

    vol = btc.volatility

    elapsed = time.time() - start

    print(f"✅ Calculation completed in {elapsed:.3f} seconds")
    print(f"   Volatility shape: {vol.shape}")
    print(f"   Mean volatility: {vol.nanmean():.4f}")
    print(f"   Std volatility: {vol[~torch.isnan(vol)].std():.4f}")
    print(f"   Min/Max: {vol[~torch.isnan(vol)].min():.4f} / {vol[~torch.isnan(vol)].max():.4f}")

    # Test multiple calls (simulating training loop)
    print(f"\nTesting 10 repeated calls (simulating training)...")
    times = []
    for i in range(10):
        start = time.time()
        _ = btc.volatility
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    print(f"✅ Average time per call: {avg_time:.4f} seconds")
    print(f"   Min: {min(times):.4f}s, Max: {max(times):.4f}s")

    if avg_time < 0.1:
        print(f"\n✅ Performance is EXCELLENT (< 0.1s per call)")
    elif avg_time < 1.0:
        print(f"\n⚠️  Performance is acceptable but could be better")
    else:
        print(f"\n❌ Performance is TOO SLOW for training")

    print("=" * 80)


if __name__ == "__main__":
    test_performance()
