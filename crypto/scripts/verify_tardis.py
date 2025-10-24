#!/usr/bin/env python3
"""
Tardis Data Verification Script

Tests and verifies data fetching from Tardis.dev API:
- Instruments metadata (options, futures, perpetuals)
- Historical trades data
- Data quality and completeness

Usage:
    # Basic verification
    python crypto/scripts/verify_tardis.py

    # With specific date
    python crypto/scripts/verify_tardis.py --date "2025-10-22 12:00"

    # Test specific instrument
    python crypto/scripts/verify_tardis.py --instrument BTC-31OCT25-108000-C --date "2025-10-22 12:00"
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto.data.tardis_client import TardisClient, timestamp_to_ms
from crypto.data.client_factory import add_client_args, create_client


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def verify_instruments(client: TardisClient, currency: str = "BTC") -> bool:
    """Verify instruments API is working."""
    print_section("TEST 1: Instruments API")

    print(f"\nFetching {currency} options...")
    try:
        instruments = client.get_instruments(currency=currency, kind="option")
        print(f"✓ Success: Found {len(instruments)} options")

        if not instruments:
            print("✗ ERROR: No instruments returned")
            return False

        # Check data structure
        sample = instruments[0]
        required_fields = [
            "instrument_name",
            "strike",
            "option_type",
            "expiration_timestamp",
        ]
        missing = [f for f in required_fields if f not in sample]

        if missing:
            print(f"✗ ERROR: Missing fields: {missing}")
            return False

        print(f"✓ Data structure valid")

        # Show samples
        print("\nSample instruments:")
        for inst in instruments[:5]:
            expiry_dt = datetime.fromtimestamp(
                inst["expiration_timestamp"] / 1000, tz=timezone.utc
            )
            print(
                f"  - {inst['instrument_name']}: "
                f"strike=${inst['strike']:,.0f}, "
                f"type={inst['option_type']}, "
                f"expiry={expiry_dt.date()}"
            )

        # Check expiry range
        expiries = sorted([inst["expiration_timestamp"] for inst in instruments])
        earliest = datetime.fromtimestamp(expiries[0] / 1000, tz=timezone.utc)
        latest = datetime.fromtimestamp(expiries[-1] / 1000, tz=timezone.utc)

        print(f"\nExpiry range:")
        print(f"  Earliest: {earliest.date()}")
        print(f"  Latest: {latest.date()}")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_spot_price(client: TardisClient, test_date: datetime) -> Optional[float]:
    """Verify spot price fetching."""
    print_section("TEST 2: Spot Price (BTC-PERPETUAL)")

    print(f"\nFetching spot price at {test_date}")

    # Get trades in ±5 minute window
    window_minutes = 5
    start = test_date - timedelta(minutes=window_minutes)
    end = test_date + timedelta(minutes=window_minutes)

    try:
        trades = client.get_historical_trades(
            instrument_name="BTC-PERPETUAL",
            start_timestamp=timestamp_to_ms(start),
            end_timestamp=timestamp_to_ms(end),
            count=100,
        )

        if not trades:
            print(f"✗ ERROR: No trades found")
            print(
                f"  This may be because data for {test_date.date()} is not yet available"
            )
            print(f"  Tardis has ~1-2 hour delay for recent data")
            return None

        print(f"✓ Found {len(trades)} trades")

        # Check trade structure
        sample = trades[0]
        required_fields = ["timestamp", "price", "amount", "direction"]
        missing = [f for f in required_fields if f not in sample]

        if missing:
            print(f"✗ ERROR: Missing fields in trades: {missing}")
            return None

        print(f"✓ Trade structure valid")

        # Calculate median price
        import pandas as pd

        df = pd.DataFrame(trades)
        spot_price = df["price"].median()

        print(f"\nSpot price: ${spot_price:,.2f}")
        print(f"Price range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")

        # Show sample trades
        print("\nSample trades:")
        for trade in trades[:3]:
            ts = datetime.fromtimestamp(trade["timestamp"] / 1000, tz=timezone.utc)
            print(
                f"  - {ts.strftime('%H:%M:%S')}: "
                f"${trade['price']:,.2f} × {trade['amount']:.2f} BTC "
                f"({trade['direction']})"
            )

        return spot_price

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return None


def verify_option_trades(
    client: TardisClient, instrument_name: str, test_date: datetime
) -> bool:
    """Verify option trades fetching."""
    print_section(f"TEST 3: Option Trades - {instrument_name}")

    print(f"\nFetching trades at {test_date}")

    # Get trades in ±5 minute window
    window_minutes = 5
    start = test_date - timedelta(minutes=window_minutes)
    end = test_date + timedelta(minutes=window_minutes)

    try:
        trades = client.get_historical_trades(
            instrument_name=instrument_name,
            start_timestamp=timestamp_to_ms(start),
            end_timestamp=timestamp_to_ms(end),
            count=1000,
        )

        print(f"✓ Found {len(trades)} trades")

        if not trades:
            print(
                f"  Note: This is normal if the option had no trades in this time window"
            )
            print(f"  Try a different instrument or time period")
            return True  # Not an error, just no liquidity

        # Check trade structure
        sample = trades[0]
        required_fields = ["timestamp", "price", "amount", "direction"]
        missing = [f for f in required_fields if f not in sample]

        if missing:
            print(f"✗ ERROR: Missing fields: {missing}")
            return False

        print(f"✓ Trade structure valid")

        # Analyze trades
        import pandas as pd

        df = pd.DataFrame(trades)

        print(f"\nTrade statistics:")
        print(f"  Premium range: {df['price'].min():.4f} - {df['price'].max():.4f} BTC")
        print(f"  Median premium: {df['price'].median():.4f} BTC")
        print(f"  Total volume: {df['amount'].sum():.2f} contracts")

        # Show sample trades
        print("\nSample trades:")
        for trade in trades[:5]:
            ts = datetime.fromtimestamp(trade["timestamp"] / 1000, tz=timezone.utc)
            print(
                f"  - {ts.strftime('%H:%M:%S')}: "
                f"{trade['price']:.4f} BTC × {trade['amount']:.2f} "
                f"({trade['direction']})"
            )

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_expiry_filtering(client: TardisClient, expiry_date: datetime) -> bool:
    """Verify filtering by expiry date works."""
    print_section("TEST 4: Expiry Date Filtering")

    expiry_str = expiry_date.strftime("%d%b%y").upper()
    print(f"\nFetching options expiring on {expiry_date.date()} ({expiry_str})")

    try:
        instruments = client.get_instruments(
            currency="BTC",
            kind="option",
            expiry_date=expiry_date,
        )

        print(f"✓ Found {len(instruments)} options for {expiry_str}")

        if not instruments:
            print(f"  Note: No options found for this expiry")
            print(
                f"  This may be because the expiry is too far in the past (beyond retention)"
            )
            return True  # Not an error

        # Verify all have correct expiry
        expiry_timestamp = int(expiry_date.timestamp() * 1000)
        tolerance_ms = 24 * 60 * 60 * 1000  # 1 day tolerance

        mismatched = [
            inst
            for inst in instruments
            if abs(inst["expiration_timestamp"] - expiry_timestamp) > tolerance_ms
        ]

        if mismatched:
            print(f"✗ ERROR: {len(mismatched)} instruments have wrong expiry")
            return False

        print(f"✓ All instruments have correct expiry")

        # Show strike distribution
        strikes = sorted(set(inst["strike"] for inst in instruments))
        print(f"\nAvailable strikes: {len(strikes)}")
        print(f"  Range: ${strikes[0]:,.0f} - ${strikes[-1]:,.0f}")

        # Count by type
        calls = sum(1 for inst in instruments if inst["option_type"] == "call")
        puts = sum(1 for inst in instruments if inst["option_type"] == "put")
        print(f"  Calls: {calls}, Puts: {puts}")

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_funding_rates(client: TardisClient, test_date: datetime) -> bool:
    """Verify funding rate history fetching."""
    print_section("TEST 5: Funding Rate History")

    print(f"\nFetching funding rates around {test_date}")

    # Get funding rates for a 24-hour period
    start = test_date - timedelta(hours=12)
    end = test_date + timedelta(hours=12)

    try:
        funding_rates = client.get_funding_rate_history(
            instrument_name="BTC-PERPETUAL",
            start_timestamp=timestamp_to_ms(start),
            end_timestamp=timestamp_to_ms(end),
        )

        print(f"✓ Found {len(funding_rates)} funding rate records")

        if not funding_rates:
            print(f"  Note: No funding rates found in this time window")
            print(f"  Funding rates are published every 8 hours on Deribit")
            return True  # Not an error, just no data in window

        # Check data structure
        sample = funding_rates[0]
        required_fields = ["timestamp", "interest_8h"]
        missing = [f for f in required_fields if f not in sample]

        if missing:
            print(f"✗ ERROR: Missing fields: {missing}")
            return False

        print(f"✓ Data structure valid")

        # Show funding rate statistics
        import pandas as pd

        df = pd.DataFrame(funding_rates)

        print(f"\nFunding rate statistics:")
        print(f"  Count: {len(df)}")
        print(f"  Mean: {df['interest_8h'].mean():.6f}")
        print(f"  Range: {df['interest_8h'].min():.6f} - {df['interest_8h'].max():.6f}")

        # Show sample records
        print("\nSample funding rates:")
        for record in funding_rates[:3]:
            ts = datetime.fromtimestamp(record["timestamp"] / 1000, tz=timezone.utc)
            rate = record.get("interest_8h", 0)
            annualized = rate * 365 * 3  # 3 times per day
            print(
                f"  - {ts.strftime('%Y-%m-%d %H:%M')}: "
                f"{rate:.6f} (annualized: {annualized*100:.2f}%)"
            )

        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_verification(
    test_date: Optional[datetime] = None,
    instrument: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """Run all verification tests."""

    # Use yesterday if no date specified (to avoid data availability issues)
    if test_date is None:
        test_date = datetime.now(timezone.utc) - timedelta(days=1)
        test_date = test_date.replace(hour=12, minute=0, second=0, microsecond=0)

    print("\n" + "=" * 80)
    print("  TARDIS DATA VERIFICATION")
    print("=" * 80)
    print(f"\nTest date: {test_date}")
    if api_key:
        print(f"API key: {api_key[:20]}...")
    else:
        print("API key: Using environment variable or free tier")

    # Create client
    try:
        client = TardisClient(api_key=api_key)
        print("✓ Client initialized")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize client: {e}")
        return False

    # Run tests
    results = []

    # Test 1: Instruments API
    results.append(("Instruments API", verify_instruments(client)))

    # Test 2: Spot price
    spot_price = verify_spot_price(client, test_date)
    results.append(("Spot Price", spot_price is not None))

    # Test 3: Option trades
    if instrument:
        results.append(
            (
                f"Option Trades ({instrument})",
                verify_option_trades(client, instrument, test_date),
            )
        )
    else:
        # Find a suitable instrument from available options
        try:
            # Get options expiring soon
            future_date = datetime.now(timezone.utc) + timedelta(days=7)
            future_date = future_date.replace(hour=8, minute=0, second=0, microsecond=0)

            instruments = client.get_instruments(
                currency="BTC",
                kind="option",
                expiry_date=future_date,
            )

            if instruments and spot_price:
                # Find ATM call
                atm_options = sorted(
                    [inst for inst in instruments if inst["option_type"] == "call"],
                    key=lambda x: abs(x["strike"] - spot_price),
                )

                if atm_options:
                    test_instrument = atm_options[0]["instrument_name"]
                    results.append(
                        (
                            f"Option Trades ({test_instrument})",
                            verify_option_trades(client, test_instrument, test_date),
                        )
                    )
        except:
            pass

    # Test 4: Expiry filtering
    expiry_date = datetime.now(timezone.utc) + timedelta(days=7)
    expiry_date = expiry_date.replace(hour=8, minute=0, second=0, microsecond=0)
    results.append(("Expiry Filtering", verify_expiry_filtering(client, expiry_date)))

    # Test 5: Funding rates
    results.append(("Funding Rate History", verify_funding_rates(client, test_date)))

    # Summary
    print_section("VERIFICATION SUMMARY")

    print(f"\nResults:")
    passed = 0
    failed = 0

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n✓ All tests passed! Tardis API is working correctly.")
        return True
    else:
        print(f"\n✗ {failed} test(s) failed. Check errors above.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify Tardis.dev data fetching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--date",
        help="Test date and time (YYYY-MM-DD HH:MM). Default: yesterday at noon",
    )
    parser.add_argument(
        "--instrument",
        help="Specific instrument to test (e.g., BTC-31OCT25-108000-C)",
    )
    parser.add_argument(
        "--api-key",
        help="Tardis API key (or set TARDIS_API_KEY env var)",
    )

    args = parser.parse_args()

    # Parse date
    test_date = None
    if args.date:
        try:
            test_date = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
        except ValueError as e:
            print(f"Error parsing date: {e}")
            return 1

    # Run verification
    success = run_verification(
        test_date=test_date,
        instrument=args.instrument,
        api_key=args.api_key,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
