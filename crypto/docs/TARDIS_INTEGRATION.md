# Tardis.dev Integration Guide

## Overview

This document describes the integration of Tardis.dev as an alternative historical data source for the crypto deep hedging module. The implementation provides seamless interoperability between Deribit's live API and Tardis.dev's comprehensive historical data (since 2019-03-30).

## Architecture

### Abstract Base Class

All market data clients implement the `MarketDataClient` abstract base class (`crypto/data/base_client.py`), which defines the standard interface:

```python
from crypto.data.base_client import MarketDataClient

class MarketDataClient(ABC):
    @abstractmethod
    def get_instruments(currency, kind, **kwargs) -> List[Dict]

    @abstractmethod
    def get_historical_trades(instrument_name, start_timestamp, end_timestamp, count) -> List[Dict]

    @abstractmethod
    def get_ticker(instrument_name, timestamp=None) -> Dict

    @abstractmethod
    def get_funding_rate_history(instrument_name, start_timestamp, end_timestamp) -> List[Dict]

    @abstractmethod
    def get_recent_trades(instrument_name, count) -> List[Dict]
```

### Client Implementations

#### DeribitClient (`crypto/data/deribit_client.py`)
- **Purpose**: Live Deribit REST API access
- **Historical coverage**: ~24 hours (limited by free API)
- **Best for**: Recent data, live trading, testing
- **Networks**: Testnet and mainnet

#### TardisClient (`crypto/data/tardis_client.py`)
- **Purpose**: Historical data from Tardis.dev
- **Historical coverage**: 2019-03-30 onwards
- **Best for**: Backtesting, historical analysis
- **Networks**: Mainnet only (historical data)
- **Free tier**: First day of each month without API key

### Client Factory

The `client_factory` module (`crypto/data/client_factory.py`) provides unified client creation:

```python
from crypto.data.client_factory import create_client

# Create Deribit client
deribit_client = create_client("deribit", testnet=True)

# Create Tardis client
tardis_client = create_client("tardis", tardis_api_key="your_key")

# Use environment variable
import os
os.environ["TARDIS_API_KEY"] = "your_key"
tardis_client = create_client("tardis")
```

## Updated Scripts

All data fetching and option discovery scripts now support both data sources:

### 1. `fetch_deribit_data.py`

Fetch perpetual trades, funding rates, and OHLC data from either source:

```bash
# Fetch from Deribit (recent data)
python crypto/scripts/fetch_deribit_data.py \
    --start 2024-01-01 --end 2024-01-02 \
    --output-dir data/historical

# Fetch from Tardis (historical data)
python crypto/scripts/fetch_deribit_data.py \
    --data-source tardis \
    --tardis-api-key YOUR_API_KEY \
    --start 2024-10-01 --end 2024-10-15 \
    --output-dir data/historical
```

### 2. `explore_options.py`

Explore available options at a specific trade date:

```bash
# Using Deribit (recent data)
python crypto/scripts/explore_options.py \
    --trade-date "2024-12-01 12:00" \
    --expiry 2024-12-15 \
    --type call

# Using Tardis (historical data)
python crypto/scripts/explore_options.py \
    --data-source tardis \
    --tardis-api-key YOUR_API_KEY \
    --trade-date "2024-10-01 12:00" \
    --expiry 2024-10-15 \
    --type call \
    --output options_oct.json
```

### 3. `select_option.py`

Select specific option and discover executed premiums:

```bash
# Using Deribit
python crypto/scripts/select_option.py \
    --trade-date "2024-12-01 12:00" \
    --expiry 2024-12-15 \
    --type call

# Using Tardis
python crypto/scripts/select_option.py \
    --data-source tardis \
    --tardis-api-key YOUR_API_KEY \
    --trade-date "2024-10-01 12:00" \
    --expiry 2024-10-15 \
    --type call \
    --output option_metadata.json
```

## Data Format Compatibility

Both clients return identical data structures, ensuring drop-in replacement:

### Trades Format
```python
{
    "timestamp": 1609459200000,  # milliseconds
    "trade_id": "123456",
    "price": 29000.5,
    "amount": 100.0,
    "direction": "buy",  # or "sell"
    "instrument_name": "BTC-PERPETUAL"
}
```

### Ticker Format
```python
{
    "instrument_name": "BTC-PERPETUAL",
    "timestamp": 1609459200000,
    "last_price": 29000.0,
    "best_bid_price": 28999.5,
    "best_ask_price": 29000.5,
    "mark_price": 29000.0,
    "index_price": 29001.0
}
```

### Funding Rate Format
```python
{
    "timestamp": 1609459200000,
    "instrument_name": "BTC-PERPETUAL",
    "interest_8h": 0.0001,
    "index_price": 29000.0
}
```

## Tardis-Specific Considerations

### 1. Instruments List

Tardis doesn't provide a real-time instruments API. `TardisClient.get_instruments()` fetches from Deribit's public API and caches the result for 1 hour.

### 2. Ticker Reconstruction

`TardisClient.get_ticker()` reconstructs ticker data from historical trades around the requested timestamp. The `timestamp` parameter is **required** for TardisClient.

```python
# TardisClient requires timestamp
ticker = tardis_client.get_ticker(
    "BTC-PERPETUAL",
    timestamp=timestamp_to_ms(datetime(2024, 10, 1, 12, 0, 0, tzinfo=timezone.utc))
)

# DeribitClient ignores timestamp (returns current data)
ticker = deribit_client.get_ticker("BTC-PERPETUAL")
```

### 3. Testnet Support

Tardis only has mainnet historical data. The `testnet=True` parameter is ignored with a warning.

### 4. Recent Trades

`get_recent_trades()` returns the last available historical trades (typically yesterday's data) rather than live trades.

## Installation

### Dependencies

Add Tardis client to your environment:

```bash
pip install tardis-client
```

Or add to `requirements.txt`:
```
tardis-client>=1.0.0
```

### API Key Setup

1. Sign up at [Tardis.dev](https://tardis.dev/)
2. Obtain API key from dashboard
3. Set environment variable:

```bash
export TARDIS_API_KEY="your_api_key_here"
```

Or pass directly:
```bash
python script.py --tardis-api-key YOUR_KEY
```

## Testing

### Unit Tests

Test TardisClient implementation:

```bash
python -m pytest crypto/tests/test_tardis_client.py -v
```

### Integration Tests

Test client interoperability:

```bash
python -m pytest crypto/tests/test_client_interop.py -v
```

### All Tests

Run complete test suite:

```bash
python -m pytest crypto/tests/ -v
```

## Example Workflow: October 2025 Backtest

Now you can backtest with October 1st 2025 data (free on Tardis):

```bash
# 1. Explore available options
python crypto/scripts/explore_options.py \
    --data-source tardis \
    --tardis-api-key $TARDIS_API_KEY \
    --trade-date "2025-10-01 12:00" \
    --expiry 2025-10-15 \
    --type call \
    --output options_oct2025.json

# 2. Select best option
python crypto/scripts/select_option.py \
    --data-source tardis \
    --tardis-api-key $TARDIS_API_KEY \
    --trade-date "2025-10-01 12:00" \
    --expiry 2025-10-15 \
    --type call \
    --output option_metadata_oct2025.json

# 3. Fetch underlying data
python crypto/scripts/fetch_deribit_data.py \
    --data-source tardis \
    --tardis-api-key $TARDIS_API_KEY \
    --start 2025-10-01 --end 2025-10-15 \
    --output-dir data/oct2025

# 4. Run backtest with fetched data
python -m crypto.backtest --config backtest_oct2025.yaml
```

## Backward Compatibility

All existing code continues to work without changes:

```bash
# Old way (still works)
python crypto/scripts/fetch_deribit_data.py \
    --start 2024-12-01 --end 2024-12-02 \
    --testnet

# This uses DeribitClient by default
```

## Implementation Files

### New Files
- `crypto/data/base_client.py` - Abstract interface
- `crypto/data/tardis_client.py` - Tardis implementation
- `crypto/data/client_factory.py` - Unified client creation
- `crypto/tests/test_tardis_client.py` - Unit tests
- `crypto/tests/test_client_interop.py` - Integration tests

### Modified Files
- `crypto/data/deribit_client.py` - Inherits from MarketDataClient
- `crypto/scripts/fetch_deribit_data.py` - Added `--data-source` flag
- `crypto/scripts/explore_options.py` - Added `--data-source` flag
- `crypto/scripts/select_option.py` - Added `--data-source` flag

## Advantages

✅ **Historical depth**: Access data since 2019-03-30 vs Deribit's ~24h
✅ **Drop-in replacement**: Same interface, same data format
✅ **Seamless switching**: Change `--data-source` flag, nothing else
✅ **Type safety**: Abstract base class enforces compatibility
✅ **Free tier**: First day of each month available without API key
✅ **Backward compatible**: Existing code works unchanged

## Troubleshooting

### ImportError: tardis-client not installed

```bash
pip install tardis-client
```

### ValueError: Tardis API key required

```bash
export TARDIS_API_KEY="your_key"
# Or pass via --tardis-api-key flag
```

### Empty trades returned

Check date range - Tardis data starts 2019-03-30. Use first of month for free access.

### "Ticker requires timestamp" error

TardisClient needs explicit timestamp for historical ticker:

```python
# Wrong
ticker = client.get_ticker("BTC-PERPETUAL")

# Correct
ticker = client.get_ticker("BTC-PERPETUAL", timestamp=timestamp_to_ms(dt))
```

## Support

- **Tardis Documentation**: https://docs.tardis.dev/
- **Tardis Python Client**: https://github.com/tardis-dev/tardis-python
- **Student Discounts**: Contact Tardis with .edu email
- **Free Trial**: Request trial account for testing

## Future Enhancements

Potential improvements:
- Add Tardis WebSocket streaming for real-time data
- Implement caching layer for frequently accessed historical data
- Add support for other Tardis channels (liquidations, options chains)
- Create hybrid client that uses Deribit for recent + Tardis for historical
