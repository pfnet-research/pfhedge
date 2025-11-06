#!/usr/bin/env python3

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

from crypto.data.base_client import MarketDataClient
from crypto.data.client_factory import create_client, add_client_args
from crypto.data.deribit_client import timestamp_to_ms, ms_to_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_available_options(
    client: MarketDataClient, expiry_date: datetime, currency: str = "BTC"
) -> List[Dict]:
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
    client: MarketDataClient, sale_time: datetime, instrument: str = "BTC-PERPETUAL"
) -> Optional[float]:
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
    client: MarketDataClient,
    option_name: str,
    check_time: datetime,
    min_trades: int = 10,
    window_hours: int = 24,
) -> Tuple[bool, int, Optional[float]]:
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
    client: MarketDataClient,
    option_name: str,
    sale_time: datetime,
    direction: str = "sell",
) -> Optional[Dict]:
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
    client: MarketDataClient,
    trade_date: datetime,
    expiry_date: datetime,
    option_type: str = "call",
    target_moneyness: float = 1.0,
    min_trades: int = 10,
) -> Optional[Dict]:
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

    # Add common client arguments (data source, testnet, API keys)
    add_client_args(parser)

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
    logger.info(f"Data source: {args.data_source}")

    # Initialize client
    try:
        client = create_client(
            data_source=args.data_source,
            testnet=args.testnet,
            tardis_api_key=args.tardis_api_key,
        )
    except (ValueError, ImportError) as e:
        logger.error(f"Failed to create client: {e}")
        sys.exit(1)

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
