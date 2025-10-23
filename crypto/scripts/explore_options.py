#!/usr/bin/env python3
"""
Option Discovery Tool

Discovers all tradeable options at a given date and expiry, showing:
- Available strikes and premiums
- Liquidity metrics (trade count)
- Implied volatilities
- Moneyness levels

Key Features:
- Full-day querying: Searches entire trading day (00:00-23:59) for maximum coverage
- Historical data access: Uses Tardis.dev API to access options back to 2019
- Fallback mode: Generates strikes when Instruments API has no data (>1 month old)
- Fast queries: ~2 minutes for ATM options (±5%), ~8 minutes for wide range (±30%)

Usage Examples:

    # Quick ATM search (recommended for most use cases)
    python crypto/scripts/explore_options.py \\
        --trade-date 2025-06-07 \\
        --expiry 2025-06-13 \\
        --type call \\
        --min-trades 5 \\
        --moneyness-range 0.95 1.05 \\
        --data-source tardis \\
        --output atm_calls.json

    # Wide search for all available options
    python crypto/scripts/explore_options.py \\
        --trade-date 2025-06-07 \\
        --expiry 2025-06-13 \\
        --type call \\
        --min-trades 1 \\
        --moneyness-range 0.7 1.3 \\
        --data-source tardis \\
        --output all_calls.json

    # With specific time (if needed)
    python crypto/scripts/explore_options.py \\
        --trade-date "2025-06-07 14:30" \\
        --expiry 2025-06-13 \\
        --type put \\
        --data-source tardis

Date Format:
    - trade-date: YYYY-MM-DD (defaults to 12:00 UTC) or "YYYY-MM-DD HH:MM"
    - expiry: YYYY-MM-DD (defaults to 08:00 UTC, Deribit expiry time) or "YYYY-MM-DD HH:MM"

How It Works:
    1. Fetches spot price from ±5 min window around trade-date for accuracy
    2. Queries Instruments API for available options at expiry
    3. If no instruments found (historical data), generates candidate strikes
    4. For each strike, queries full trading day (00:00-23:59) for trades
    5. Calculates premiums, IV, and filters by liquidity (min-trades)

Performance:
    - ATM (moneyness 0.95-1.05): ~11 strikes, ~2 minutes
    - Wide (moneyness 0.7-1.3): ~40 strikes, ~7-8 minutes
    - Per-strike query: ~10-11 seconds (network-bound)

Deribit Option Types:
    - Daily: Expire every day at 08:00 UTC (48h after listing)
    - Weekly: Expire every Friday at 08:00 UTC
    - Monthly: Last Friday of month at 08:00 UTC
    - Quarterly: Last Friday of Mar/Jun/Sep/Dec at 08:00 UTC

Requirements:
    - Tardis.dev API key (set TARDIS_API_KEY environment variable)
    - Historical data available back to 2019
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

from crypto.data.base_client import MarketDataClient
from crypto.data.client_factory import create_client, add_client_args
from crypto.utils.black_scholes import implied_volatility_from_btc_premium

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_strikes(
    spot_price: float, moneyness_range: tuple = (0.8, 1.2)
) -> List[int]:
    """
    Generate candidate strikes matching Deribit's grid pattern.

    Deribit uses different strike intervals depending on price level:
    - 1K intervals around ATM (±10% from spot)
    - 2K intervals for medium OTM (±20% from spot)
    - 5K intervals for the full range

    Args:
        spot_price: Current spot price
        moneyness_range: (min, max) moneyness to consider

    Returns:
        Sorted list of candidate strike prices
    """
    min_price = spot_price * moneyness_range[0]
    max_price = spot_price * moneyness_range[1]

    strikes = set()

    # 1. Dense ATM grid: 1K intervals within ±10% of spot
    atm_min = int(spot_price * 0.90 / 1000) * 1000
    atm_max = int(spot_price * 1.10 / 1000) * 1000 + 1000

    strike = atm_min
    while strike <= atm_max:
        if min_price <= strike <= max_price:
            strikes.add(strike)
        strike += 1000

    # 2. Medium density: 2K intervals within ±20%
    med_min = int(spot_price * 0.80 / 2000) * 2000
    med_max = int(spot_price * 1.20 / 2000) * 2000 + 2000

    strike = med_min
    while strike <= med_max:
        if min_price <= strike <= max_price:
            strikes.add(strike)
        strike += 2000

    # 3. Sparse grid: 5K intervals for full range
    sparse_min = int(min_price / 5000) * 5000
    sparse_max = int(max_price / 5000) * 5000 + 5000

    strike = sparse_min
    while strike <= sparse_max:
        if min_price <= strike <= max_price:
            strikes.add(strike)
        strike += 5000

    return sorted(strikes)


def _build_instrument_name(expiry_date: datetime, strike: int, option_type: str) -> str:
    """
    Build Deribit instrument name following their convention.

    Format: BTC-{D}MMMYY}-{STRIKE}-{C|P} (no leading zero on day)
    Example: BTC-5SEP25-110000-C (not BTC-05SEP25-110000-C)

    Args:
        expiry_date: Option expiry date
        strike: Strike price
        option_type: 'call' or 'put'

    Returns:
        Instrument name string
    """
    # Deribit doesn't use leading zeros for single-digit days
    day = expiry_date.day
    month = expiry_date.strftime("%b").upper()
    year = expiry_date.strftime("%y")
    expiry_str = f"{day}{month}{year}"

    type_char = "C" if option_type == "call" else "P"
    return f"BTC-{expiry_str}-{strike}-{type_char}"


def get_available_options(
    client: MarketDataClient,
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

    # Get a narrow window around trade_date for spot price (±5 minutes)
    from datetime import timedelta
    from crypto.data.deribit_client import timestamp_to_ms

    window_minutes = 5
    spot_start = trade_date - timedelta(minutes=window_minutes)
    spot_end = trade_date + timedelta(minutes=window_minutes)

    try:
        trades = client.get_historical_trades(
            instrument_name="BTC-PERPETUAL",
            start_timestamp=timestamp_to_ms(spot_start),
            end_timestamp=timestamp_to_ms(spot_end),
            count=100,
        )

        if not trades:
            logger.error("No perpetual trades found at trade date")
            return []

        # Convert to DataFrame for easier processing
        import pandas as pd

        trades_df = pd.DataFrame(trades)
        initial_spot = trades_df["price"].median()
        logger.info(f"Initial spot price: ${initial_spot:,.2f}")

    except Exception as e:
        logger.error(f"Error fetching spot price: {e}")
        return []

    # Create full-day window for option trade queries
    # This captures all trades on the trade date, maximizing liquidity discovery
    day_start = trade_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = trade_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    # Get available options for expiry
    expiry_str = expiry_date.strftime("%d%b%y").upper()
    logger.info(f"Querying options for expiry: {expiry_str}")

    # Get active instruments filtered by expiry date
    instruments = client.get_instruments(
        currency="BTC",
        kind="option",
        expired=False,
        expiry_date=expiry_date,
    )

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

    # Fallback: If no instruments found, generate strikes and query directly
    if len(matching_instruments) == 0:
        logger.warning(
            f"No instruments found in API for {expiry_str}. "
            f"This may be because the expiry is beyond the API retention period (~1 month). "
            f"Falling back to strike generation and direct trade queries."
        )

        # Generate candidate strikes based on spot price and moneyness range
        strikes = generate_strikes(initial_spot, moneyness_range)
        logger.info(f"Generated {len(strikes)} candidate strikes to query")

        # Build synthetic instrument list for consistency with main flow
        matching_instruments = [
            {
                "instrument_name": _build_instrument_name(
                    expiry_date, strike, option_type
                ),
                "strike": strike,
            }
            for strike in strikes
        ]

        logger.info(
            f"Fallback mode: will query {len(matching_instruments)} candidate instruments"
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
            # Get trades from the full trade_date (00:00 to 23:59)
            # Using full-day window maximizes finding trades for historical options
            trades = client.get_historical_trades(
                instrument_name=instrument_name,
                start_timestamp=timestamp_to_ms(day_start),
                end_timestamp=timestamp_to_ms(day_end),
                count=1000,
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

            # Format IV for logging
            iv_str = f"{iv:.1%}" if iv else "N/A"
            logger.info(
                f"  ✓ Premium: {premium_btc:.4f} BTC (${premium_usd:,.2f}), "
                f"IV: {iv_str}, Trades: {len(trades)}"
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
        help="Trade date: YYYY-MM-DD (defaults to 12:00 UTC) or 'YYYY-MM-DD HH:MM'",
    )
    parser.add_argument(
        "--expiry",
        required=True,
        help="Expiry date: YYYY-MM-DD (defaults to 08:00 UTC) or 'YYYY-MM-DD HH:MM'",
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
    parser.add_argument("--output", "-o", help="Output JSON file path (optional)")

    # Add common client arguments (data source, testnet, API keys)
    add_client_args(parser)

    args = parser.parse_args()

    # Parse dates - accept both date (YYYY-MM-DD) and datetime (YYYY-MM-DD HH:MM)
    try:
        # Parse trade_date - if only date provided, default to noon (12:00) for spot price
        trade_date_str = args.trade_date.strip()
        if len(trade_date_str) == 10:  # Just date: YYYY-MM-DD
            trade_date = datetime.fromisoformat(trade_date_str).replace(
                hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
            )
        else:  # Full datetime
            trade_date = datetime.fromisoformat(trade_date_str).replace(
                tzinfo=timezone.utc
            )

        # Parse expiry_date - if only date provided, use 08:00 UTC (Deribit expiry time)
        expiry_date_str = args.expiry.strip()
        if len(expiry_date_str) == 10:  # Just date: YYYY-MM-DD
            expiry_date = datetime.fromisoformat(expiry_date_str).replace(
                hour=8, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
            )
        else:  # Full datetime
            expiry_date = datetime.fromisoformat(expiry_date_str).replace(
                tzinfo=timezone.utc
            )
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Use format: YYYY-MM-DD or YYYY-MM-DD HH:MM")
        return 1

    # Validate date order
    if trade_date >= expiry_date:
        print("Error: Trade date must be before expiry date")
        return 1

    # Create market data client
    try:
        client = create_client(
            data_source=args.data_source,
            testnet=args.testnet,
            tardis_api_key=args.tardis_api_key,
        )
    except (ValueError, ImportError) as e:
        print(f"Error creating client: {e}")
        return 1

    print("\n" + "=" * 120)
    print("OPTION EXPLORATION")
    print("=" * 120)
    print(f"\nParameters:")
    print(f"  Data source: {args.data_source}")
    print(f"  Trade date: {trade_date}")
    print(f"  Expiry: {expiry_date}")
    print(f"  Type: {args.type}")
    print(f"  Min trades: {args.min_trades}")
    print(
        f"  Moneyness range: {args.moneyness_range[0]:.2f} - {args.moneyness_range[1]:.2f}"
    )
    if args.data_source == "deribit":
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
