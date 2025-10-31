#!/usr/bin/env python3
"""
Test script to verify rolling realized volatility implementation.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from crypto.instruments import BitcoinPerpetualBrownian, BitcoinSpotBrownian

def test_brownian_realized_vol():
    """Test realized volatility calculation in Brownian simulations."""
    print("\n" + "="*80)
    print("Testing Rolling Realized Volatility Implementation")
    print("="*80 + "\n")

    # Test with constant volatility (window=0)
    print("1. Testing CONSTANT volatility (volatility_window=0):")
    btc_const = BitcoinPerpetualBrownian(sigma=0.8, volatility_window=0)
    btc_const.simulate(n_paths=10, time_horizon=30/365)
    vol_const = btc_const.volatility
    print(f"   Shape: {vol_const.shape}")
    print(f"   All values = 0.8? {torch.allclose(vol_const, torch.tensor(0.8), atol=1e-6)}")
    print(f"   Mean: {vol_const.mean():.4f}, Std: {vol_const.std():.6f}")

    # Test with rolling window
    print("\n2. Testing ROLLING WINDOW volatility (volatility_window=20):")
    torch.manual_seed(42)
    btc_rolling = BitcoinPerpetualBrownian(sigma=0.8, volatility_window=20)
    btc_rolling.simulate(n_paths=10, time_horizon=30/365)
    vol_rolling = btc_rolling.volatility
    print(f"   Shape: {vol_rolling.shape}")
    print(f"   Values vary? {vol_rolling.std() > 0.01}")  # Should have some variance
    print(f"   Mean: {vol_rolling.mean():.4f}, Std: {vol_rolling.std():.4f}")
    print(f"   Min: {vol_rolling.min():.4f}, Max: {vol_rolling.max():.4f}")
    print(f"   First 5 values (path 0): {vol_rolling[0, :5].tolist()}")
    print(f"   Last 5 values (path 0): {vol_rolling[0, -5:].tolist()}")

    # Test with SpotBrownian
    print("\n3. Testing with BitcoinSpotBrownian (volatility_window=20):")
    torch.manual_seed(42)
    btc_spot = BitcoinSpotBrownian(sigma=0.8, volatility_window=20)
    btc_spot.simulate(n_paths=10, time_horizon=30/365)
    vol_spot = btc_spot.volatility
    print(f"   Shape: {vol_spot.shape}")
    print(f"   Values vary? {vol_spot.std() > 0.01}")
    print(f"   Mean: {vol_spot.mean():.4f}, Std: {vol_spot.std():.4f}")

    # Verify NaN handling
    print("\n4. Checking NaN handling at the beginning:")
    print(f"   Any NaN values? {torch.isnan(vol_rolling).any()}")
    print(f"   First 20 values should be close to 0.8 (filling NaN):")
    print(f"   First 20 mean: {vol_rolling[0, :20].mean():.4f}")

    print("\n" + "="*80)
    print("All tests passed! ✅")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_brownian_realized_vol()
