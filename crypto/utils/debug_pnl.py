#!/usr/bin/env python3
"""
Debug script to compare PnL calculations.

This verifies that our manual BS PnL calculation matches PFHedge's native method.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
from crypto.instruments import create_bitcoin_option_from_config
from crypto.strategies import create_deep_hedger, calculate_bs_hedge_pnl
from pfhedge.nn.functional import cum_pl


def compare_pnl_calculations():
    """Compare our manual BS PnL with PFHedge's native cum_pl."""

    print("=" * 70)
    print("PNL CALCULATION COMPARISON")
    print("=" * 70)

    # Create a simple test case
    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        "strike": 50000,
        "maturity_days": 14,
        "volatility": 0.8,
        "drift": 0.0,
        "cost": 0.0005,
        "call": True,
        "cost": 0.0,
        "sigma": 0.8,
        "mu": 0.0,
        "underlier_cost": 0.0005,
        "n_paths": 10,
        "seed": 42,
    }

    option, _ = create_bitcoin_option_from_config(config)

    # Get spot prices and calculate BS delta
    spots = option.underlier.spot  # Shape: (n_paths, n_steps)
    bs_delta = option.black_scholes_delta()  # Shape: (n_paths, n_steps)
    payoffs = option.payoff()  # Shape: (n_paths,)
    cost = config["underlier_cost"]

    print(f"\nSpot shape: {spots.shape}")
    print(f"BS delta shape: {bs_delta.shape}")
    print(f"Payoffs shape: {payoffs.shape}")

    # Method 1: Our manual calculation
    print("\n" + "-" * 70)
    print("METHOD 1: Our Manual Calculation")
    print("-" * 70)

    bs_hedge_pnl_manual = calculate_bs_hedge_pnl(spots, bs_delta, payoffs, cost)

    print(f"PnL shape: {bs_hedge_pnl_manual.shape}")
    print(f"Final PnL: {bs_hedge_pnl_manual[:, -1]}")
    print(f"Mean final PnL: {bs_hedge_pnl_manual[:, -1].mean().item():.2f}")
    print(f"Std final PnL: {bs_hedge_pnl_manual[:, -1].std().item():.2f}")

    # Method 2: PFHedge's native cum_pl
    print("\n" + "-" * 70)
    print("METHOD 2: PFHedge Native cum_pl")
    print("-" * 70)

    # Need to reshape for PFHedge: (N, H, T) where H=1 for single instrument
    spots_pfhedge = spots.unsqueeze(1)  # (n_paths, 1, n_steps)
    units_pfhedge = bs_delta.unsqueeze(1)  # (n_paths, 1, n_steps)

    bs_hedge_pnl_native = cum_pl(
        spot=spots_pfhedge,
        unit=units_pfhedge,
        cost=[cost],
        payoff=payoffs,
        deduct_first_cost=True,
    )

    print(f"PnL shape: {bs_hedge_pnl_native.shape}")
    print(f"Final PnL: {bs_hedge_pnl_native[:, -1]}")
    print(f"Mean final PnL: {bs_hedge_pnl_native[:, -1].mean().item():.2f}")
    print(f"Std final PnL: {bs_hedge_pnl_native[:, -1].std().item():.2f}")

    # Method 3: Deep Hedger (for comparison)
    print("\n" + "-" * 70)
    print("METHOD 3: Deep Hedger compute_cum_pl")
    print("-" * 70)

    hedger = create_deep_hedger(n_layers=3, n_units=64)
    # Train minimally
    hedger.fit(option, n_paths=100, n_epochs=5, verbose=False)

    with torch.no_grad():
        deep_pnl = hedger.compute_cum_pl(option).squeeze()

    print(f"PnL shape: {deep_pnl.shape}")
    print(f"Final PnL: {deep_pnl[:, -1]}")
    print(f"Mean final PnL: {deep_pnl[:, -1].mean().item():.2f}")
    print(f"Std final PnL: {deep_pnl[:, -1].std().item():.2f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    diff_manual_native = torch.abs(bs_hedge_pnl_manual - bs_hedge_pnl_native)
    max_diff = diff_manual_native.max().item()
    mean_diff = diff_manual_native.mean().item()

    print(f"\nManual vs Native:")
    print(f"  Max difference: ${max_diff:.2f}")
    print(f"  Mean difference: ${mean_diff:.2f}")
    print(f"  Match: {'✅ YES' if max_diff < 1.0 else '❌ NO'}")

    if max_diff >= 1.0:
        print("\n⚠️  SIGNIFICANT DIFFERENCE DETECTED!")
        print("Our manual BS PnL calculation likely has a bug.")
        print("\nDetailed comparison of first path:")
        print(f"  Manual:  {bs_hedge_pnl_manual[0, -5:]}")
        print(f"  Native:  {bs_hedge_pnl_native[0, -5:]}")

        # Check intermediate steps
        print("\nDebugging first path:")
        print(f"  Spot prices (last 5): {spots[0, -5:]}")
        print(f"  BS delta (last 5): {bs_delta[0, -5:]}")
        print(f"  Payoff: {payoffs[0]}")
    else:
        print("\n✅ Manual calculation matches PFHedge native method!")

    print("\n" + "=" * 70)

    return {
        "manual": bs_hedge_pnl_manual,
        "native": bs_hedge_pnl_native,
        "deep": deep_pnl,
        "max_diff": max_diff,
        "match": max_diff < 1.0,
    }


if __name__ == "__main__":
    results = compare_pnl_calculations()

    if not results["match"]:
        print("\n❌ BUG FOUND IN MANUAL PNL CALCULATION!")
        print("This explains why deep hedging appears to perform poorly.")
        print("The BS baseline PnL is incorrectly calculated.")
    else:
        print("\n🤔 Manual PnL calculation is correct.")
        print("The issue must be elsewhere...")
