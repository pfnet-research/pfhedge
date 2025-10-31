#!/usr/bin/env python3
"""
Verify Strike Normalization in Saved Checkpoint

Checks that a trained model checkpoint has the correct normalized strike,
not the absolute strike. This helps catch training bugs where strike
normalization was skipped.

Usage:
    python crypto/scripts/verify_checkpoint.py models/my_model/model.pth

    # With expected strike value
    python crypto/scripts/verify_checkpoint.py \
        models/my_model/model.pth \
        --expected-strike 0.9604
"""

import argparse
import sys
from pathlib import Path

import torch


def verify_checkpoint(checkpoint_path: str, expected_strike: float = None) -> bool:
    """
    Verify that checkpoint contains normalized strike.

    Args:
        checkpoint_path: Path to model.pth checkpoint
        expected_strike: Optional expected strike value for verification

    Returns:
        True if checkpoint appears valid, False otherwise
    """
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

    # Check structure
    if "training_config" not in checkpoint:
        print(f"❌ Checkpoint missing 'training_config' key")
        return False

    config = checkpoint["training_config"]

    # Extract strike
    if "strike" not in config:
        print(f"❌ Training config missing 'strike' key")
        return False

    strike = config["strike"]

    # Verify strike is normalized (should be close to 1.0, typically 0.8-1.2)
    print(f"\n📊 Checkpoint Verification: {checkpoint_path}")
    print(f"   Strike: {strike:.6f}")

    # Check if strike looks normalized
    if strike < 0.5 or strike > 2.0:
        print(f"   ⚠️  WARNING: Strike {strike:.6f} is outside typical range [0.5, 2.0]")
        print(f"   This might indicate an unnormalized absolute strike!")
        is_valid = False
    else:
        print(f"   ✅ Strike appears normalized (in range [0.5, 2.0])")
        is_valid = True

    # Additional checks
    if strike > 10000:
        print(f"   ❌ Strike {strike:.0f} looks like absolute strike (e.g., $105,000)")
        print(f"   Training was likely done with WRONG strike!")
        return False

    # Check against expected value if provided
    if expected_strike is not None:
        diff = abs(strike - expected_strike)
        rel_diff = diff / expected_strike if expected_strike != 0 else float("inf")

        print(f"\n🎯 Expected Strike: {expected_strike:.6f}")
        print(f"   Absolute difference: {diff:.6f}")
        print(f"   Relative difference: {rel_diff:.2%}")

        if rel_diff > 0.01:  # 1% tolerance
            print(
                f"   ❌ Strike differs from expected by {rel_diff:.2%} (threshold: 1%)"
            )
            is_valid = False
        else:
            print(f"   ✅ Strike matches expected value within 1%")

    # Print other config details for context
    print(f"\n📋 Other Config Details:")
    print(f"   Maturity days: {config.get('maturity_days', 'N/A')}")
    print(f"   Option type: {'Call' if config.get('call', True) else 'Put'}")
    print(
        f"   Volatility: {config.get('volatility', 'N/A'):.1%}"
        if isinstance(config.get("volatility"), (int, float))
        else f"   Volatility: {config.get('volatility', 'N/A')}"
    )
    print(
        f"   Transaction cost: {config.get('transaction_cost', 'N/A'):.2%}"
        if isinstance(config.get("transaction_cost"), (int, float))
        else f"   Transaction cost: {config.get('transaction_cost', 'N/A')}"
    )

    return is_valid


def main():
    parser = argparse.ArgumentParser(
        description="Verify strike normalization in model checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "checkpoint",
        help="Path to model checkpoint (model.pth)",
    )
    parser.add_argument(
        "--expected-strike",
        type=float,
        help="Expected normalized strike value for verification",
    )

    args = parser.parse_args()

    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"\n❌ Error: Checkpoint not found: {args.checkpoint}\n")
        return 1

    # Run verification
    is_valid = verify_checkpoint(str(checkpoint_path), args.expected_strike)

    # Summary
    print("\n" + "=" * 70)
    if is_valid:
        print("✅ CHECKPOINT VERIFICATION PASSED")
        print("=" * 70 + "\n")
        return 0
    else:
        print("❌ CHECKPOINT VERIFICATION FAILED")
        print("=" * 70)
        print(
            "\nThe checkpoint may have been trained with incorrect strike normalization."
        )
        print("Consider retraining the model with the fixed train_for_option.py.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
