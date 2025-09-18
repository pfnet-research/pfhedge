# Revised Deep Hedging Engine Plan - Quick MVP in One Week

## Goal: Working Backtest in 7 Days
Focus: Historical data, backtesting with PFHedge, no real-time components

## Prerequisites
- **Deribit Account**: Sign up for free account to access historical data API
- **API Keys**: Get read-only API credentials
- **Python 3.9+** with PyTorch and PFHedge installed

## Day 1-2: Data Pipeline (Get Historical Data Working)

### 1.1 Minimal Deribit Client
**File**: `crypto/data/deribit_client.py`
- REST API client for historical data only
- Methods: get_historical_trades(), get_historical_options()
- Use requests library, no WebSocket needed

### 1.2 Data Downloader Script
**File**: `crypto/data/download_historical.py`
- Download 3 months of BTC perpetual trades
- Download ATM options data for same period
- Save as Parquet files locally
- Progress bar with tqdm

### 1.3 Quick Data Loader
**File**: `crypto/data/loader.py`
- Load Parquet files into pandas DataFrames
- Resample to 5-minute bars
- Calculate returns, volatility
- Align timestamps across instruments

## Day 3-4: Extend PFHedge for Crypto

### 2.1 Bitcoin Primary Instrument
**File**: `crypto/instruments/bitcoin_spot.py`
```python
class BitcoinSpot(pfhedge.instruments.Primary):
    # Extend Primary class
    # Load historical prices from our data
    # Include funding rate as attribute
```

### 2.2 Bitcoin Option Instrument
**File**: `crypto/instruments/bitcoin_option.py`
```python
class BitcoinOption(pfhedge.instruments.EuropeanOption):
    # Extend EuropeanOption
    # Adapt for inverse contracts (BTC denomination)
    # Use actual historical IV from Deribit
```

### 2.3 Market Friction Features
**File**: `crypto/features/frictions.py`
- Create custom features for PFHedge:
  - Historical bid-ask spread feature
  - Funding rate feature
  - Simple market impact model

## Day 5: Implement Hedging Strategies

### 3.1 Delta Hedge Baseline
**File**: `crypto/strategies/delta_hedge.py`
```python
class DeltaHedgeStrategy(pfhedge.nn.modules.BSModuleMixin):
    # Use PFHedge's Black-Scholes modules
    # Add no-trade band (threshold rebalancing)
    # Include transaction costs
```

### 3.2 Deep Hedging Model
**File**: `crypto/strategies/deep_hedge.py`
```python
# Use pfhedge.nn.Hedger directly
hedger = pfhedge.nn.Hedger(
    model=pfhedge.nn.MLP(),  # Start with simple MLP
    inputs=["log_moneyness", "time_to_maturity", "volatility"],
    criterion="mean_variance"  # Will switch to CVaR later
)
```

## Day 6: Backtesting Engine

### 4.1 Simple Backtest Runner
**File**: `crypto/backtest/runner.py`
- Load historical data
- Create BitcoinOption with historical prices
- Run hedger.fit() on training period
- Run hedger.price() on test period
- Calculate PnL with transaction costs

### 4.2 Metrics Calculator
**File**: `crypto/backtest/metrics.py`
- PnL after costs
- CVaR (5% and 1%)
- Sharpe ratio
- Max drawdown
- Use pfhedge.nn.functional for risk measures

## Day 7: Run Experiments & Report

### 5.1 Training Script
**File**: `crypto/train.py`
```python
# Quick training script
# Load 2 months data for training
# Train both delta hedge and deep hedge
# Save models
```

### 5.2 Evaluation Script
**File**: `crypto/evaluate.py`
```python
# Load saved models
# Run backtest on 1 month holdout data
# Compare strategies side-by-side
# Generate plots with matplotlib
```

### 5.3 Results Notebook
**File**: `crypto/results.ipynb`
- Jupyter notebook with results
- PnL curves comparison
- Risk metrics table
- Trade frequency analysis

## Quick Win Implementation Order

1. **Start with data** (Day 1-2)
   - Get Deribit historical data working first
   - You can see real Bitcoin data immediately

2. **Leverage PFHedge** (Day 3-4)
   - Minimal custom code, maximum reuse
   - Focus on adapters, not reimplementing

3. **Use existing models** (Day 5)
   - Start with pfhedge.nn.Hedger and MLP
   - Don't build custom architectures yet

4. **Simple backtest** (Day 6)
   - One instrument, one strategy at a time
   - Get metrics working with small dataset first

5. **Iterate quickly** (Day 7)
   - Run experiments
   - Document what works

## What We're NOT Doing (Yet)
- ❌ Real-time data feeds
- ❌ Live trading
- ❌ Complex market microstructure
- ❌ Multiple strikes/expirations
- ❌ Order book simulation
- ❌ Limit orders
- ❌ Paper trading

## Success Criteria for Week 1
✅ Downloaded 3 months of Bitcoin historical data
✅ Trained a deep hedging model using PFHedge
✅ Backtested against delta hedge baseline
✅ Generated comparison report with CVaR metrics
✅ Working code that others can run

## Next Steps (After Week 1)
- Add more sophisticated features
- Implement proper CVaR objective
- Expand to multiple strikes
- Add market impact models
- Build proper evaluation framework

This plan gets you a working system in 7 days by maximizing use of PFHedge's existing capabilities and focusing only on what's needed for backtesting.