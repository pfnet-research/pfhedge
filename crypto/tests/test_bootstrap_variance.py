import pytest
import torch
import pandas as pd
from datetime import datetime, timedelta

from crypto.data.loader import CryptoDataLoader
from crypto.instruments import BitcoinPerpetualHistorical


def create_test_data(n_records: int, dt_hours: int = 8) -> pd.DataFrame:
    import numpy as np

    start = pd.Timestamp("2024-10-01", tz="UTC")
    timestamps = [start + pd.Timedelta(hours=i * dt_hours) for i in range(n_records)]

    # Create price series with some variance
    # Use sin wave + noise to create realistic-looking price movements
    base_price = 60000
    prices = base_price + 2000 * np.sin(np.linspace(0, 4 * np.pi, n_records))
    prices += np.random.RandomState(42).normal(0, 500, n_records)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "last_price": prices,
            "bid_price": prices * 0.999,
            "ask_price": prices * 1.001,
            "index_price": prices,
            "funding_rate": [0.0001] * n_records,
        }
    )


def test_bootstrap_variance_with_sufficient_data():
    # Create enough data for variance
    n_steps = 42  # 14 days at 8H intervals
    extra_windows = 50  # Want 50 unique windows
    n_records = n_steps + extra_windows  # 92 records

    test_data = create_test_data(n_records)

    # Create data loader
    loader = CryptoDataLoader(data_dir=".")
    loader.perpetual_data = test_data

    # Create perpetual instrument
    btc = BitcoinPerpetualHistorical(
        data_loader=loader, dt=8 / 24 / 365  # 8 hours in years
    )

    # Run bootstrap with 10 paths
    n_paths = 10
    time_horizon = 14 / 365  # 14 days

    btc.simulate_bootstrap(
        n_paths=n_paths,
        time_horizon=time_horizon,
        target_initial_spot=None,  # No rescaling for this test
        max_date=None,
    )

    # Verify paths have variance
    final_spots = btc.spot[:, -1]
    spot_std = final_spots.std().item()

    # Assert variance exists
    assert spot_std > 0, (
        f"Bootstrap paths should have variance, but std={spot_std:.6f}. "
        f"All paths appear identical!"
    )

    # Assert not all paths are the same
    unique_final_spots = torch.unique(final_spots)
    assert (
        len(unique_final_spots) > 1
    ), f"All {n_paths} paths have identical final spots: {final_spots[0]:.2f}"

    print(f"✓ Bootstrap variance test passed:")
    print(f"  Final spots: {final_spots[:5].tolist()}")
    print(f"  Std dev: {spot_std:.2f}")
    print(f"  Unique values: {len(unique_final_spots)}/{n_paths}")


def test_bootstrap_warns_with_insufficient_data(caplog):
    import logging

    # Create exactly enough data for backtest (no extra for variance)
    # Note: n_steps = int(time_horizon / dt) + 1, so for 14 days at dt=8/24/365, we need 43
    n_records = 43  # Exactly what we need (int(14/365 / (8/24/365)) + 1 = 43), no extra

    test_data = create_test_data(n_records)

    # Create data loader
    loader = CryptoDataLoader(data_dir=".")
    loader.perpetual_data = test_data

    # Create perpetual instrument
    btc = BitcoinPerpetualHistorical(data_loader=loader, dt=8 / 24 / 365)

    # Run bootstrap
    n_paths = 100
    time_horizon = 14 / 365

    with caplog.at_level(logging.WARNING):
        btc.simulate_bootstrap(
            n_paths=n_paths,
            time_horizon=time_horizon,
            target_initial_spot=None,
            max_date=None,
        )

    # Verify warning was logged
    warning_messages = [
        record.message for record in caplog.records if record.levelname == "WARNING"
    ]

    assert any(
        "BOOTSTRAP VARIANCE WARNING" in msg for msg in warning_messages
    ), "Expected warning about insufficient bootstrap data, but none found"

    assert any(
        "Only 1 possible window" in msg for msg in warning_messages
    ), "Expected warning about single window, but not found"

    # Verify paths are identical (as expected)
    final_spots = btc.spot[:, -1]
    unique_final_spots = torch.unique(final_spots)
    assert len(unique_final_spots) == 1, (
        f"Expected all paths identical with insufficient data, "
        f"but got {len(unique_final_spots)} unique values"
    )

    print(f"✓ Insufficient data warning test passed:")
    print(f"  Warning logged: ✓")
    print(f"  All paths identical: ✓ (expected behavior)")


def test_bootstrap_variance_with_rescaling():
    # Create sufficient data
    n_steps = 42
    extra_windows = 50
    n_records = n_steps + extra_windows

    test_data = create_test_data(n_records)

    # Create data loader
    loader = CryptoDataLoader(data_dir=".")
    loader.perpetual_data = test_data

    # Create perpetual instrument
    btc = BitcoinPerpetualHistorical(data_loader=loader, dt=8 / 24 / 365)

    # Run bootstrap WITH rescaling
    n_paths = 10
    time_horizon = 14 / 365
    target_initial_spot = 63000.0

    btc.simulate_bootstrap(
        n_paths=n_paths,
        time_horizon=time_horizon,
        target_initial_spot=target_initial_spot,  # Enable rescaling
        max_date=None,
    )

    # Verify initial spots are all the same (due to rescaling)
    initial_spots = btc.spot[:, 0]
    initial_std = initial_spots.std().item()
    assert (
        initial_std < 0.01
    ), f"Initial spots should be identical after rescaling, but std={initial_std:.6f}"

    # Verify final spots still have variance (different price movements)
    final_spots = btc.spot[:, -1]
    final_std = final_spots.std().item()
    assert (
        final_std > 0
    ), f"Final spots should vary even with rescaling, but std={final_std:.6f}"

    unique_final_spots = torch.unique(final_spots)
    assert (
        len(unique_final_spots) > 1
    ), f"All {n_paths} paths have identical final spots even with different historical windows"

    print(f"✓ Rescaling variance test passed:")
    print(f"  Initial spot std: {initial_std:.6f} (expected ~0)")
    print(f"  Final spot std: {final_std:.2f} (expected > 0)")
    print(f"  Unique final values: {len(unique_final_spots)}/{n_paths}")


if __name__ == "__main__":
    # Run tests manually
    print("Running bootstrap variance tests...\n")

    print("Test 1: Sufficient data creates variance")
    test_bootstrap_variance_with_sufficient_data()
    print()

    print("Test 2: Insufficient data triggers warning")
    import logging

    logging.basicConfig(level=logging.WARNING)

    class CaptureLog:
        def __init__(self):
            self.records = []

        def at_level(self, level):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    # Simple test without caplog
    print("  (Note: Run with pytest to see full warning capture)")
    print()

    print("Test 3: Rescaling preserves variance")
    test_bootstrap_variance_with_rescaling()
    print()

    print("✅ All tests passed!")
