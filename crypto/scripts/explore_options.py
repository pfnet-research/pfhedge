#!/usr/bin/env python3
"""
Option Discovery Tool

Explores available options at a given date and expiry, showing:
- Available strikes and premiums
- Liquidity metrics
- Implied volatilities
- Moneyness levels

This allows traders to review options before selecting one to trade.

Usage:
    # Explore options for Oct 29 expiry, trading on Oct 15
    python crypto/scripts/explore_options.py \
        --trade-date "2024-10-15 12:00" \
        --expiry 2024-10-29 \
        --type call \
        --output options_candidates.json

    # With custom filters
    python crypto/scripts/explore_options.py \
        --trade-date "2024-10-15 12:00" \
        --expiry 2024-10-29 \
        --type call \
        --min-trades 10 \
        --moneyness-range 0.9 1.1 \
        --testnet \
        --output options_candidates.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto.data.deribit_client import DeribitClient
from crypto.utils.black_scholes import implied_volatility_from_btc_premium

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_available_options(
    client: DeribitClient,
    trade_date: datetime,
    expiry_date: datetime,
    option_type: str = "call",
    min_trades: int = 1,
    moneyness_range: tuple = (0.8, 1.2),
) -> List[Dict]:
    """
    Get all available options matching criteria.

    Args:
        client: Deribit client
        trade_date: When option would be traded
        expiry_date: Option expiry date
        option_type: 'call' or 'put'
        min_trades: Minimum number of trades for liquidity
        moneyness_range: (min, max) moneyness to consider

    Returns:
        List of option dictionaries with metadata
    """
    # Get perpetual price at trade date
    logger.info(f"Fetching spot price at {trade_date}")

    # Get a narrow window around trade_date for spot price
    from crypto.scripts.fetch_deribit_data import fetch_perpetual_trades

    start = trade_date
    end = trade_date

    trades_df = fetch_perpetual_trades(client, start, end, instrument="BTC-PERPETUAL")

    if trades_df.empty:
        logger.error("No perpetual trades found at trade date")
        return []

    initial_spot = trades_df["price"].median()
    logger.info(f"Initial spot price: ${initial_spot:,.2f}")

    # Get available options for expiry
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    logger.info(f"Querying options for expiry: {expiry_str}")

    instruments = client.get_instruments(currency="BTC", kind="option", expired=False)

    # Filter by expiry and type
    matching_instruments = [
        inst
        for inst in instruments
        if expiry_str in inst["instrument_name"]
        and (
            "-C" in inst["instrument_name"]
            if option_type == "call"
            else "-P" in inst["instrument_name"]
        )
    ]

    logger.info(
        f"Found {len(matching_instruments)} {option_type} options for {expiry_str}"
    )

    # Calculate moneyness and filter
    min_moneyness, max_moneyness = moneyness_range

    options = []
    for inst in matching_instruments:
        strike = inst["strike"]
        moneyness = initial_spot / strike

        # Filter by moneyness
        if not (min_moneyness <= moneyness <= max_moneyness):
            continue

        instrument_name = inst["instrument_name"]

        # Get trade history to check liquidity and premium
        logger.info(f"Checking {instrument_name} (K={strike}, M={moneyness:.3f})")

        try:
            trades = client.get_last_trades_by_instrument(
                instrument_name=instrument_name, count=1000, include_old=True
            )

            if not trades or len(trades) < min_trades:
                logger.info(
                    f"  Skipping - insufficient trades ({len(trades) if trades else 0})"
                )
                continue

            # Calculate premium statistics from trades
            import pandas as pd

            trades_df = pd.DataFrame(trades)

            # Use 25th percentile for conservative seller pricing (bid-side)
            premium_btc = trades_df["price"].quantile(0.25)
            premium_usd = premium_btc * initial_spot

            # Calculate implied volatility
            days_to_expiry = (expiry_date - trade_date).days
            time_to_expiry = days_to_expiry / 365.0

            iv = implied_volatility_from_btc_premium(
                premium_btc=premium_btc,
                spot=initial_spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                option_type=option_type,
            )

            option_info = {
                "instrument_name": instrument_name,
                "strike": strike,
                "option_type": option_type,
                "initial_spot": initial_spot,
                "moneyness": moneyness,
                "premium_btc": premium_btc,
                "premium_usd": premium_usd,
                "implied_volatility": iv if iv else None,
                "trade_count": len(trades),
                "days_to_expiry": days_to_expiry,
                "trade_date": trade_date.isoformat(),
                "expiry_date": expiry_date.isoformat(),
            }

            options.append(option_info)

            logger.info(
                f"  ✓ Premium: {premium_btc:.4f} BTC (${premium_usd:,.2f}), "
                f"IV: {iv:.1%} if iv else 'N/A', Trades: {len(trades)}"
            )

        except Exception as e:
            logger.warning(f"  Error fetching trades for {instrument_name}: {e}")
            continue

    # Sort by moneyness (ATM first)
    options.sort(key=lambda x: abs(x["moneyness"] - 1.0))

    return options


def print_options_table(options: List[Dict]):
    """Print options in a readable table format."""
    if not options:
        print("\nNo options found matching criteria.")
        return

    print("\n" + "=" * 120)
    print("AVAILABLE OPTIONS")
    print("=" * 120)
    print(
        f"\n{'Instrument':<25} {'Strike':>10} {'Moneyness':>10} {'Premium (BTC)':>15} "
        f"{'Premium (USD)':>15} {'IV':>8} {'Trades':>8}"
    )
    print("-" * 120)

    for opt in options:
        iv_str = (
            f"{opt['implied_volatility']:.1%}" if opt["implied_volatility"] else "N/A"
        )
        print(
            f"{opt['instrument_name']:<25} "
            f"${opt['strike']:>9,.0f} "
            f"{opt['moneyness']:>10.3f} "
            f"{opt['premium_btc']:>15.4f} "
            f"${opt['premium_usd']:>14,.2f} "
            f"{iv_str:>8} "
            f"{opt['trade_count']:>8,}"
        )

    print("=" * 120)
    print(f"\nTotal options found: {len(options)}")
    print(f"Initial spot price: ${options[0]['initial_spot']:,.2f}")
    print(f"Trade date: {options[0]['trade_date']}")
    print(f"Expiry date: {options[0]['expiry_date']}")
    print(f"Days to expiry: {options[0]['days_to_expiry']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Explore available options for trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--trade-date",
        required=True,
        help="Date and time to trade option (YYYY-MM-DD HH:MM)",
    )
    parser.add_argument(
        "--expiry", required=True, help="Option expiry date (YYYY-MM-DD)"
    )

    # Optional arguments
    parser.add_argument(
        "--type",
        choices=["call", "put"],
        default="call",
        help="Option type (default: call)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="Minimum number of trades for liquidity (default: 10)",
    )
    parser.add_argument(
        "--moneyness-range",
        nargs=2,
        type=float,
        default=[0.9, 1.1],
        metavar=("MIN", "MAX"),
        help="Moneyness range to consider (default: 0.9 1.1)",
    )
    parser.add_argument(
        "--testnet", action="store_true", help="Use Deribit testnet instead of mainnet"
    )
    parser.add_argument("--output", "-o", help="Output JSON file path (optional)")

    args = parser.parse_args()

    # Parse dates
    try:
        trade_date = datetime.fromisoformat(args.trade_date).replace(
            tzinfo=timezone.utc
        )
        expiry_date = datetime.fromisoformat(args.expiry).replace(tzinfo=timezone.utc)
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        return 1

    # Validate date order
    if trade_date >= expiry_date:
        print("Error: Trade date must be before expiry date")
        return 1

    # Create Deribit client
    client = DeribitClient(testnet=args.testnet)

    print("\n" + "=" * 120)
    print("OPTION EXPLORATION")
    print("=" * 120)
    print(f"\nParameters:")
    print(f"  Trade date: {trade_date}")
    print(f"  Expiry: {expiry_date}")
    print(f"  Type: {args.type}")
    print(f"  Min trades: {args.min_trades}")
    print(
        f"  Moneyness range: {args.moneyness_range[0]:.2f} - {args.moneyness_range[1]:.2f}"
    )
    print(f"  Network: {'Testnet' if args.testnet else 'Mainnet'}")
    print()

    # Get available options
    try:
        options = get_available_options(
            client=client,
            trade_date=trade_date,
            expiry_date=expiry_date,
            option_type=args.type,
            min_trades=args.min_trades,
            moneyness_range=tuple(args.moneyness_range),
        )
    except Exception as e:
        print(f"\nError fetching options: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Print results
    print_options_table(options)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "search_parameters": {
                        "trade_date": trade_date.isoformat(),
                        "expiry_date": expiry_date.isoformat(),
                        "option_type": args.type,
                        "min_trades": args.min_trades,
                        "moneyness_range": args.moneyness_range,
                        "testnet": args.testnet,
                    },
                    "options": options,
                },
                f,
                indent=2,
                default=str,
            )

        print(f"✓ Results saved to: {output_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
