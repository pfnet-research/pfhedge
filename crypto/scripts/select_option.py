#!/usr/bin/env python3
"""
Select options from historical Deribit data and discover executed premiums.

This script:
1. Finds available options for a target expiry
2. Calculates ATM relative to underlying price
3. Filters for liquidity
4. Fetches actual executed premiums from historical trades

Usage:
    python select_option.py --trade-date 2024-01-15 --expiry 2024-01-29 --output metadata.json
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

# Add parent directory to path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from crypto.data.deribit_client import DeribitClient, timestamp_to_ms, ms_to_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_available_options(
    client: DeribitClient, expiry_date: datetime, currency: str = "BTC"
) -> List[Dict]:
    """
    Get all available options for a specific expiry date.

    Args:
        client: Deribit client
        expiry_date: Target expiry date
        currency: Currency (BTC or ETH)

    Returns:
        List of option instruments
    """
    logger.info(
        f"Fetching available {currency} options for expiry {expiry_date.date()}"
    )

    try:
        instruments = client.get_instruments(currency=currency, kind="option")

        # Filter by expiry date
        target_expiry_ms = timestamp_to_ms(expiry_date)
        matching_options = []

        for inst in instruments:
            if inst.get("expiration_timestamp") == target_expiry_ms:
                matching_options.append(inst)

        logger.info(f"Found {len(matching_options)} options for target expiry")
        return matching_options

    except Exception as e:
        logger.error(f"Error fetching instruments: {e}")
        return []


def get_underlying_price(
    client: DeribitClient, sale_time: datetime, instrument: str = "BTC-PERPETUAL"
) -> Optional[float]:
    """
    Get underlying price at a specific time.

    Args:
        client: Deribit client
        sale_time: Time to get price
        instrument: Underlying instrument (perpetual or index)

    Returns:
        Price at the specified time, or None if not found
    """
    logger.info(f"Fetching {instrument} price at {sale_time}")

    try:
        # Fetch trades around the sale time
        window_minutes = 5
        start_time = sale_time - timedelta(minutes=window_minutes)
        end_time = sale_time + timedelta(minutes=window_minutes)

        trades = client.get_historical_trades(
            instrument_name=instrument,
            start_timestamp=timestamp_to_ms(start_time),
            end_timestamp=timestamp_to_ms(end_time),
            count=100,
        )

        if not trades:
            logger.warning(f"No trades found for {instrument} at {sale_time}")
            return None

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Find trade closest to sale time
        df["time_diff"] = abs((df["timestamp"] - sale_time).dt.total_seconds())
        closest_trade = df.loc[df["time_diff"].idxmin()]

        price = closest_trade["price"]
        logger.info(f"Underlying price at {sale_time}: ${price:.2f}")

        return price

    except Exception as e:
        logger.error(f"Error fetching underlying price: {e}")
        return None


def calculate_moneyness(spot: float, strike: float, option_type: str = "call") -> float:
    """Calculate moneyness (spot/strike for calls, strike/spot for puts)."""
    if option_type.lower() == "call":
        return spot / strike
    else:
        return strike / spot


def select_atm_options(
    options: List[Dict],
    spot_price: float,
    target_moneyness: float = 1.0,
    tolerance: float = 0.05,
) -> List[Dict]:
    """
    Select ATM options based on moneyness.

    Args:
        options: List of option instruments
        spot_price: Current spot price
        target_moneyness: Target moneyness (1.0 for ATM)
        tolerance: Tolerance for moneyness matching

    Returns:
        List of options close to target moneyness
    """
    atm_options = []

    for opt in options:
        strike = opt.get("strike")
        if not strike:
            continue

        # Determine option type from instrument name
        inst_name = opt.get("instrument_name", "")
        option_type = "call" if "-C" in inst_name else "put"

        moneyness = calculate_moneyness(spot_price, strike, option_type)

        if abs(moneyness - target_moneyness) <= tolerance:
            opt["moneyness"] = moneyness
            atm_options.append(opt)

    # Sort by how close to target moneyness
    atm_options.sort(key=lambda x: abs(x["moneyness"] - target_moneyness))

    logger.info(f"Found {len(atm_options)} options near ATM")

    return atm_options


def check_option_liquidity(
    client: DeribitClient,
    option_name: str,
    check_time: datetime,
    min_trades: int = 10,
    window_hours: int = 24,
) -> Tuple[bool, int, Optional[float]]:
    """
    Check if option has sufficient liquidity.

    Args:
        client: Deribit client
        option_name: Option instrument name
        check_time: Time to check liquidity
        min_trades: Minimum number of trades required
        window_hours: Hours to look back for trades

    Returns:
        Tuple of (is_liquid, trade_count, average_price)
    """
    logger.debug(f"Checking liquidity for {option_name}")

    try:
        start_time = check_time - timedelta(hours=window_hours)
        trades = client.get_historical_trades(
            instrument_name=option_name,
            start_timestamp=timestamp_to_ms(start_time),
            end_timestamp=timestamp_to_ms(check_time),
            count=1000,
        )

        trade_count = len(trades)
        is_liquid = trade_count >= min_trades

        avg_price = None
        if trades:
            prices = [t["price"] for t in trades]
            avg_price = np.mean(prices)

        return is_liquid, trade_count, avg_price

    except Exception as e:
        logger.error(f"Error checking liquidity: {e}")
        return False, 0, None


def get_executed_premium(
    client: DeribitClient,
    option_name: str,
    sale_time: datetime,
    direction: str = "sell",
) -> Optional[Dict]:
    """
    Get actual executed premium from historical trades.

    Args:
        client: Deribit client
        option_name: Option instrument name
        sale_time: Time of sale
        direction: "sell" or "buy"

    Returns:
        Dictionary with premium info or None
    """
    logger.info(f"Fetching executed premium for {option_name} at {sale_time}")

    try:
        # Fetch trades around sale time
        window_minutes = 15
        trades = client.get_historical_trades(
            instrument_name=option_name,
            start_timestamp=timestamp_to_ms(
                sale_time - timedelta(minutes=window_minutes)
            ),
            end_timestamp=timestamp_to_ms(
                sale_time + timedelta(minutes=window_minutes)
            ),
            count=100,
        )

        if not trades:
            logger.warning(f"No trades found for {option_name} around {sale_time}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Find trades closest to sale time
        df["time_diff"] = abs((df["timestamp"] - sale_time).dt.total_seconds())
        df = df.sort_values("time_diff")

        # For selling, we want bid-side execution (lower prices)
        # For buying, we want ask-side execution (higher prices)
        if direction == "sell":
            # Take lower quartile of prices (conservative estimate for sell)
            premium = df["price"].quantile(0.25)
        else:
            # Take upper quartile for buy
            premium = df["price"].quantile(0.75)

        # Get additional info
        result = {
            "premium": premium,
            "btc_premium": premium,  # Already in BTC for Deribit
            "trade_count": len(df),
            "avg_price": df["price"].mean(),
            "min_price": df["price"].min(),
            "max_price": df["price"].max(),
            "closest_trade_time": df.iloc[0]["timestamp"].isoformat(),
            "time_diff_seconds": df.iloc[0]["time_diff"],
        }

        logger.info(f"Executed premium: {premium:.4f} BTC (from {len(df)} trades)")

        return result

    except Exception as e:
        logger.error(f"Error fetching executed premium: {e}")
        return None


def select_best_option(
    client: DeribitClient,
    trade_date: datetime,
    expiry_date: datetime,
    option_type: str = "call",
    target_moneyness: float = 1.0,
    min_trades: int = 10,
) -> Optional[Dict]:
    """
    Select the best option based on criteria.

    Args:
        client: Deribit client
        trade_date: Date of option trade
        expiry_date: Target expiry date
        option_type: "call" or "put"
        target_moneyness: Target moneyness (1.0 for ATM)
        min_trades: Minimum trades for liquidity

    Returns:
        Dictionary with selected option info
    """
    # Get underlying price at trade time
    initial_spot = get_underlying_price(client, trade_date)
    if not initial_spot:
        logger.error("Could not get underlying price")
        return None

    # Get available options
    options = get_available_options(client, expiry_date)
    if not options:
        logger.error("No options available for expiry")
        return None

    # Filter by option type
    type_suffix = "-C" if option_type == "call" else "-P"
    options = [o for o in options if type_suffix in o.get("instrument_name", "")]

    # Select ATM options
    atm_options = select_atm_options(options, initial_spot, target_moneyness)
    if not atm_options:
        logger.error("No ATM options found")
        return None

    # Check liquidity and select best
    best_option = None
    best_liquidity = 0

    for opt in atm_options[:5]:  # Check top 5 ATM options
        option_name = opt["instrument_name"]
        is_liquid, trade_count, avg_price = check_option_liquidity(
            client, option_name, trade_date, min_trades
        )

        if is_liquid and trade_count > best_liquidity:
            best_option = opt
            best_liquidity = trade_count

    if not best_option:
        logger.warning("No liquid options found, using closest ATM")
        best_option = atm_options[0]

    # Get executed premium
    option_name = best_option["instrument_name"]
    premium_info = get_executed_premium(client, option_name, trade_date, "sell")

    if not premium_info:
        logger.error("Could not get executed premium")
        return None

    # Calculate days to expiry
    days_to_expiry = (expiry_date - trade_date).days

    # Compile final result
    result = {
        "instrument_name": option_name,
        "strike": best_option["strike"],
        "option_type": option_type,
        "expiry_date": expiry_date.isoformat(),
        "trade_date": trade_date.isoformat(),
        "days_to_expiry": days_to_expiry,
        "initial_spot": initial_spot,
        "moneyness": calculate_moneyness(
            initial_spot, best_option["strike"], option_type
        ),
        "premium_btc": premium_info["premium"],
        "premium_usd": premium_info["premium"] * initial_spot,  # Convert to USD
        "trade_count": premium_info["trade_count"],
        "liquidity_score": best_liquidity,
        "premium_details": premium_info,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Select options and discover premiums")
    parser.add_argument(
        "--trade-date", type=str, required=True, help="Trade date (YYYY-MM-DD HH:MM)"
    )
    parser.add_argument(
        "--expiry", type=str, required=True, help="Expiry date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--type", type=str, choices=["call", "put"], default="call", help="Option type"
    )
    parser.add_argument(
        "--moneyness", type=float, default=1.0, help="Target moneyness (1.0 for ATM)"
    )
    parser.add_argument(
        "--min-trades", type=int, default=10, help="Minimum trades for liquidity"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="option_metadata.json",
        help="Output file for metadata",
    )
    parser.add_argument("--testnet", action="store_true", help="Use testnet")

    args = parser.parse_args()

    # Parse dates
    if " " in args.trade_date:
        trade_date = datetime.strptime(args.trade_date, "%Y-%m-%d %H:%M")
    else:
        trade_date = datetime.strptime(args.trade_date, "%Y-%m-%d")
    trade_date = trade_date.replace(tzinfo=timezone.utc)

    expiry_date = datetime.strptime(args.expiry, "%Y-%m-%d")
    expiry_date = expiry_date.replace(
        hour=8, tzinfo=timezone.utc
    )  # Deribit expires at 8 UTC

    logger.info(f"Selecting {args.type} option")
    logger.info(f"Trade date: {trade_date}")
    logger.info(f"Expiry: {expiry_date}")
    logger.info(f"Target moneyness: {args.moneyness}")

    # Initialize client
    client = DeribitClient(testnet=args.testnet)

    # Select best option
    result = select_best_option(
        client,
        trade_date,
        expiry_date,
        option_type=args.type,
        target_moneyness=args.moneyness,
        min_trades=args.min_trades,
    )

    if not result:
        logger.error("Failed to select option")
        sys.exit(1)

    # Save metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Saved option metadata to {output_path}")

    # Print summary
    print(f"\nSelected Option:")
    print(f"  Instrument: {result['instrument_name']}")
    print(f"  Strike: ${result['strike']:,.0f}")
    print(f"  Type: {result['option_type']}")
    print(f"  Days to expiry: {result['days_to_expiry']}")
    print(f"  Initial spot: ${result['initial_spot']:,.2f}")
    print(f"  Moneyness: {result['moneyness']:.3f}")
    print(f"  Premium (BTC): {result['premium_btc']:.4f}")
    print(f"  Premium (USD): ${result['premium_usd']:,.2f}")
    print(f"  Liquidity: {result['trade_count']} trades")


if __name__ == "__main__":
    main()
