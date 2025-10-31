#!/usr/bin/env python3
"""
Test to check which device the volatility calculation runs on.
"""

import torch
import time
from crypto.instruments.bitcoin_spot_brownian import BitcoinSpotBrownian


def test_device_performance():
    """Test performance on CPU vs GPU."""
    print("=" * 80)
    print("Volatility Calculation Device Test")
    print("=" * 80)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠️  CUDA not available, testing CPU only")

    n_paths = 5000
    time_horizon = 30/365

    for device in devices:
        print(f"\n{'='*80}")
        print(f"Testing on: {device}")
        print('='*80)

        # Create instrument on specific device
        btc = BitcoinSpotBrownian(
            sigma=0.42,
            mu=0.0,
            cost=0.0001,
            dt=8/24/365,
            volatility_window=20,
            device=device
        )

        print(f"\nSimulating {n_paths} paths...")
        btc.simulate(n_paths=n_paths, time_horizon=time_horizon)

        print(f"Spot device: {btc.spot.device}")
        print(f"Spot shape: {btc.spot.shape}")

        # Warm-up call
        _ = btc.volatility

        # Time multiple calls
        print(f"\nTiming 10 volatility calculations...")
        times = []
        for i in range(10):
            start = time.time()
            vol = btc.volatility
            if device.type == 'cuda':
                torch.cuda.synchronize()  # Wait for GPU to finish
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)

        print(f"\nResults on {device}:")
        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Min: {min(times):.4f}s, Max: {max(times):.4f}s")
        print(f"  Volatility device: {vol.device}")
        print(f"  Mean volatility: {vol.nanmean():.4f}")

    if len(devices) > 1:
        cpu_time = avg_time  # Last one was GPU if available
        # Re-run CPU for comparison
        btc_cpu = BitcoinSpotBrownian(
            sigma=0.42, mu=0.0, cost=0.0001, dt=8/24/365,
            volatility_window=20, device=torch.device("cpu")
        )
        btc_cpu.simulate(n_paths=n_paths, time_horizon=time_horizon)

        start = time.time()
        _ = btc_cpu.volatility
        cpu_time = time.time() - start

        speedup = cpu_time / avg_time if avg_time > 0 else 0
        print(f"\n{'='*80}")
        print(f"GPU Speedup: {speedup:.2f}x faster than CPU")
        print('='*80)

    print("\n" + "="*80)


if __name__ == "__main__":
    test_device_performance()
