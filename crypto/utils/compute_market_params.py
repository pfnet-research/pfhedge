#!/usr/bin/env python3
"""
Compute realistic market parameters from Deribit data.

This script fetches recent Bitcoin data from Deribit and computes:
- Current spot price
- Realized volatility (annualized)
- Drift (mu)
- Transaction costs from market data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from crypto.data.deribit_client import DeribitClient


def fetch_recent_btc_data(days=30):
    """Fetch recent Bitcoin perpetual data from Deribit.

    Args:
        days: Number of days of historical data to fetch

    Returns:
        DataFrame with timestamp, price, and other market data
    """
    print(f"Fetching {days} days of BTC perpetual data from Deribit...")

    client = DeribitClient()

    # Get current time and start time
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    # Convert to milliseconds
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    # Fetch data
    trades = []
    current_ms = start_ms

    while current_ms < end_ms:
        try:
            response = client.get_recent_trades(
                instrument_name="BTC-PERPETUAL",
                start_timestamp=current_ms,
                end_timestamp=min(current_ms + 3600000 * 24, end_ms),  # 24 hours
                count=10000
            )

            if not response or 'trades' not in response:
                break

            batch = response['trades']
            if not batch:
                break

            trades.extend(batch)

            # Update timestamp
            last_timestamp = batch[-1]['timestamp']
            current_ms = last_timestamp + 1

            print(f"  Fetched {len(batch)} trades (up to {datetime.fromtimestamp(last_timestamp/1000)})")

        except Exception as e:
            print(f"  Error fetching data: {e}")
            break

    if not trades:
        print("Warning: No trades fetched!")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp')

    print(f"\n✅ Fetched {len(df)} trades from {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def compute_realized_volatility(prices, window_hours=24, frequency='1h'):
    """Compute realized volatility from price series.

    Args:
        prices: Series of prices with datetime index
        window_hours: Rolling window size in hours
        frequency: Resampling frequency

    Returns:
        Annualized realized volatility
    """
    # Resample to regular intervals
    prices_resampled = prices.resample(frequency).last().dropna()

    # Compute log returns
    log_returns = np.log(prices_resampled / prices_resampled.shift(1)).dropna()

    # Compute rolling volatility
    rolling_vol = log_returns.rolling(window=window_hours).std()

    # Annualize (assuming 365 days, 24 hours per day)
    periods_per_year = 365 * 24 / 1  # Hourly data
    annualized_vol = rolling_vol * np.sqrt(periods_per_year)

    # Return recent average (last 7 days)
    recent_vol = annualized_vol.tail(24 * 7).mean()

    return recent_vol


def compute_drift(prices, frequency='1h'):
    """Compute drift (mu) from price series.

    Args:
        prices: Series of prices with datetime index
        frequency: Resampling frequency

    Returns:
        Annualized drift
    """
    # Resample to regular intervals
    prices_resampled = prices.resample(frequency).last().dropna()

    # Compute log returns
    log_returns = np.log(prices_resampled / prices_resampled.shift(1)).dropna()

    # Mean return
    mean_return = log_returns.mean()

    # Annualize
    periods_per_year = 365 * 24 / 1  # Hourly data
    annualized_drift = mean_return * periods_per_year

    return annualized_drift


def get_current_market_params(days=30):
    """Get current market parameters from Deribit data.

    Args:
        days: Number of days of historical data to use

    Returns:
        Dictionary with market parameters
    """
    # Fetch data
    df = fetch_recent_btc_data(days=days)

    if df is None or len(df) == 0:
        print("Failed to fetch data, using default parameters")
        return {
            'spot_price': 50000,
            'volatility': 0.8,
            'drift': 0.0,
            'cost': 0.0005,  # 0.05% from Deribit taker fee
        }

    # Create price series
    df = df.set_index('timestamp')
    prices = df['price']

    # Compute parameters
    current_price = prices.iloc[-1]
    volatility = compute_realized_volatility(prices)
    drift = compute_drift(prices)

    # Transaction cost from Deribit (taker fee)
    cost = 0.0005  # 0.05% = 5 basis points

    params = {
        'spot_price': float(current_price),
        'volatility': float(volatility),
        'drift': float(drift),
        'cost': cost,
        'data_start': df.index[0],
        'data_end': df.index[-1],
        'n_trades': len(df),
    }

    return params


def print_market_params(params):
    """Pretty print market parameters."""
    print("\n" + "="*60)
    print("CURRENT BITCOIN MARKET PARAMETERS")
    print("="*60)
    print(f"Spot Price:        ${params['spot_price']:,.2f}")
    print(f"Realized Vol:      {params['volatility']:.2%} annualized")
    print(f"Drift (mu):        {params['drift']:.2%} annualized")
    print(f"Transaction Cost:  {params['cost']:.2%} (Deribit taker fee)")

    if 'data_start' in params:
        print(f"\nData Period:")
        print(f"  Start: {params['data_start']}")
        print(f"  End:   {params['data_end']}")
        print(f"  Trades: {params['n_trades']:,}")

    print("="*60)


if __name__ == "__main__":
    # Compute and display market parameters
    params = get_current_market_params(days=30)
    print_market_params(params)

    # Save to file for use in bitcoin_hedge.py
    import json
    output_file = os.path.join(os.path.dirname(__file__), 'market_params.json')

    # Convert datetime to string for JSON serialization
    params_json = params.copy()
    if 'data_start' in params_json:
        params_json['data_start'] = str(params_json['data_start'])
    if 'data_end' in params_json:
        params_json['data_end'] = str(params_json['data_end'])

    with open(output_file, 'w') as f:
        json.dump(params_json, f, indent=2)

    print(f"\n✅ Saved parameters to {output_file}")
