# Step-by-Step Interactive Trading Workflow

Complete guide for interactive option trading with human decision points.

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [Step 0: Download Data](#step-0-download-historical-data)
- [Overview](#overview)
- [Workflow Diagram](#workflow-diagram)
- [Step 1: Explore Options](#step-1-explore-available-options)
- [Step 2: Train Model](#step-2-train-model-for-selected-option)
- [Step 3: Backtest](#step-3-backtest-the-strategy)
- [Step 4: Calculate Returns](#step-4-calculate-expected-returns)
- [Complete Example](#complete-example-walkthrough)
- [Tips for Real Trading](#tips-for-real-trading)
- [Troubleshooting](#troubleshooting)

---

## Quick Reference

### Script Comparison

| Script | Purpose | Human Decision | Output |
|--------|---------|----------------|--------|
| `explore_options.py` | Find available options | ✅ Review & select | JSON with options |
| `train_for_option.py` | Train for specific option | ✅ Review training | Model checkpoint |
| `crypto.backtest.run` | Test strategy | ✅ Review backtest | Hedging P&L |
| `calculate_seller_pnl.py` | Final returns | ✅ Final decision | Total P&L |
| `realistic_backtest.py` | Automated pipeline | ❌ No review | Everything |

### Quick Start Commands

```bash
# 1. Explore options (simplified date format)
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 \
    --expiry 2024-10-29 \
    --type call \
    --data-source tardis \
    --output options.json

# 2. Train model (after selecting instrument)
python crypto/scripts/train_for_option.py \
    --option-file options.json \
    --instrument BTC-29OCT24-50000-C \
    --output models/my_model

# 3. Backtest (create config first, see crypto/configs/backtest_example.yaml)
python -m crypto.backtest.run \
    --config my_backtest.yaml \
    --seed 42

# 4. Calculate returns
python crypto/scripts/calculate_seller_pnl.py \
    --option options.json \
    --instrument BTC-29OCT24-50000-C \
    --backtest backtest_results/results.json
```

### Step 0: Download Historical Data (Required Before Backtesting)

**IMPORTANT**: Backtest data must be downloaded before running backtests. The workflow requires two types of data:

| Data Type | Source | Method | Why |
|-----------|--------|--------|-----|
| **Perpetual OHLC** | Deribit API | `get_ohlc_candles` | Fast, pre-aggregated 8-hour bars |
| **Funding rates** | Tardis CSV + Deribit API | CSV downloads + REST API | Complete historical coverage |

**Recommended approach (hybrid data sources):**

The script automatically handles the optimal data fetching strategy:
- **OHLC data**: Fetched from Deribit's TradingView API (60-min candles → resampled to 8-hour)
- **Funding rates**:
  - Historical (>30 days): Tardis CSV downloads (fast, comprehensive)
  - Recent (<30 days): Deribit REST API (official rates)

**Command:**
```bash
# Download OHLC + funding rates for backtesting period
# Requires Tardis API key for historical funding data
python crypto/scripts/fetch_deribit_data.py \
    --data-source tardis \
    --start 2025-01-01 \
    --end 2025-10-01 \
    --instrument BTC-PERPETUAL \
    --frequency 8H \
    --output-dir crypto/data/historical \
    --tardis-api-key YOUR_API_KEY
```

**What this does:**
1. Fetches BTC-PERPETUAL OHLC candles from Deribit (automatic batching for long periods)
2. Downloads funding rates from Tardis CSV datasets (Jan-Aug historical data)
3. Downloads funding rates from Deribit API (Sept-Oct recent data)
4. Automatically merges and aligns data to 8-hour intervals
5. Saves to parquet files for fast loading during backtests

**Output files:**
```
crypto/data/historical/
├── btc-perpetual_8H_2025-01-01_2025-10-01.parquet          # OHLC price data
└── btc-perpetual_funding_complete_2025-01-01_2025-10-01.parquet  # Funding rates
```

### ⚠️  CRITICAL: Bootstrap Data Requirements

**IMPORTANT**: For meaningful bootstrap statistics with variance > 0, you MUST download MORE data than your backtest period!

**The Problem**: If you download exactly the backtest period, all bootstrap paths will be IDENTICAL (variance = 0.00).

**Formula for data requirements**:
```
recommended_records = backtest_steps + desired_unique_windows

Where:
- backtest_steps = maturity_days * 24 / dt_hours
- desired_unique_windows = max(100, n_bootstrap_paths / 10)
- dt_hours = rebalancing frequency (typically 8)
```

**Example Calculation**:
```
Backtest: Oct 1-15 (14 days)
Bootstrap paths: 1000
dt_hours: 8

backtest_steps = 14 * 24 / 8 = 42 steps
desired_windows = max(100, 1000/10) = 100 windows
recommended_records = 42 + 100 = 142 records

Days of data needed: 142 * 8 / 24 = 47 days
Download range: Sep 1 - Oct 16 (not Oct 1-15!)
```

**Correct command for 14-day backtest**:
```bash
# Download 47 days for 14-day backtest (100 unique bootstrap windows)
python crypto/scripts/fetch_deribit_data.py \
    --start 2024-09-01 \  # 33 days BEFORE backtest start!
    --end 2024-10-16 \    # 1 day after backtest end
    --output-dir crypto/data/historical
```

**What happens if you don't follow this?**
- ❌ Download Oct 1-15 only → 42 records → 1 window → ALL paths identical
- ❌ All bootstrap statistics show std = 0.00 (meaningless)
- ❌ No variance in P&L, Sharpe, or any metric
- ✅ Download Sep 1 - Oct 16 → 142 records → 100 unique windows → Valid statistics

**Quick reference**:
| Backtest Days | dt_hours | Min Extra Days | Total Download |
|---------------|----------|----------------|----------------|
| 7 days        | 8        | ~30 days       | ~37 days       |
| 14 days       | 8        | ~33 days       | ~47 days       |
| 30 days       | 8        | ~33 days       | ~63 days       |

**For testing (no API key needed, limited to first day of each month):**
```bash
# Use Tardis free tier for testing
python crypto/scripts/fetch_deribit_data.py \
    --data-source tardis \
    --start 2025-01-01 \
    --end 2025-01-03 \
    --output-dir crypto/data/historical
```

**For recent data only (no Tardis needed):**
```bash
# Deribit-only mode (last 30 days of funding available)
python crypto/scripts/fetch_deribit_data.py \
    --data-source deribit \
    --start 2025-09-01 \
    --end 2025-10-01 \
    --output-dir crypto/data/historical
```

**Why this approach?**
- **Fast**: OHLC from Deribit is 100x faster than individual trade replay
- **Accurate**: Tardis funding rates are 8-hour TWAP (same as used for settlements)
- **Complete**: Tardis has full history back to 2019; Deribit only retains 30 days
- **Reliable**: CSV downloads avoid WebSocket replay timeouts

### Supporting Scripts

- **`realistic_backtest.py`** - Automated end-to-end pipeline (for batch jobs only)
  ```bash
  python crypto/scripts/realistic_backtest.py \
      --config crypto/configs/realistic_backtest_example.yaml
  ```

---

## Overview

This workflow is designed for **real trading** where you need to:
- Review available options before selecting
- Evaluate training results before proceeding
- Analyze backtest performance before trading
- Make informed decisions at each step

**NOT for automated batch jobs** - use `realistic_backtest.py` for that.

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Download Historical Data                           │
│ Script: fetch_deribit_data.py                               │
│ Output: OHLC + funding rate parquet files                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Explore Available Options                          │
│ Script: explore_options.py                                  │
│ Output: options_candidates.json                             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
         [ HUMAN DECISION ]
         Review options, pick one
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Train Model for Selected Option                    │
│ Script: train_for_option.py                                 │
│ Output: models/MODEL_NAME/model.pth + training_results.json │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
         [ HUMAN DECISION ]
         Review training, approve model
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Backtest the Strategy                               │
│ Script: crypto.backtest.run                                  │
│ Output: backtest_results/results.json + report.md           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
         [ HUMAN DECISION ]
         Review backtest performance
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Calculate Expected Returns                          │
│ Script: calculate_seller_pnl.py                             │
│ Output: final_analysis.json                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
         [ FINAL DECISION ]
         Execute trade or pass
```

---

## Step 0: Download Historical Data

### Purpose
Download perpetual contract OHLC data and funding rates required for backtesting. This data is used to simulate realistic market conditions during strategy evaluation.

### Data Requirements

For backtesting, you need:
1. **BTC-PERPETUAL OHLC prices** (open, high, low, close at 8-hour intervals)
2. **Funding rate history** (8-hour funding payments for perpetual positions)

### Download Script

```bash
python crypto/scripts/fetch_deribit_data.py \
    --data-source tardis \
    --start 2025-01-01 \
    --end 2025-10-01 \
    --instrument BTC-PERPETUAL \
    --frequency 8H \
    --output-dir crypto/data/historical \
    --tardis-api-key YOUR_API_KEY
```

### Parameters
- `--data-source`: Use `tardis` for historical data (back to 2019), or `deribit` for recent data only
- `--start`, `--end`: Date range for data download (format: YYYY-MM-DD)
- `--instrument`: Perpetual contract name (default: BTC-PERPETUAL)
- `--frequency`: Resampling frequency (default: 8H for 8-hour intervals)
- `--output-dir`: Directory to save downloaded data
- `--tardis-api-key`: Your Tardis.dev API key (required for historical data >30 days)

### How It Works

**Data fetching strategy:**
1. **OHLC candles**: Fetched from Deribit's TradingView API
   - Downloads 60-minute candles in batches (handles API limit of ~5000 candles)
   - Automatically resamples to 8-hour intervals
   - Fast: ~2 seconds for 9 months vs ~23 minutes for individual trades

2. **Funding rates**: Hybrid approach for complete coverage
   - **Historical (>30 days ago)**: Downloads from Tardis CSV datasets
     - One CSV file per day, gzip compressed
     - Samples funding rates at 8-hour intervals (00:00, 08:00, 16:00 UTC)
     - Efficient: downloads 243 days in ~15 minutes
   - **Recent (<30 days)**: Fetches from Deribit REST API
     - Deribit only retains ~30 days of funding history
     - Provides official 8-hour rates

3. **Automatic merging**: Script combines Tardis and Deribit funding data
   - Uses Tardis for historical data
   - Uses Deribit for recent data
   - Removes duplicates and aligns timestamps

### Output Files

```
crypto/data/historical/
├── btc-perpetual_8H_2025-01-01_2025-10-01.parquet
│   └── OHLC price data (823 bars, 100% coverage)
└── btc-perpetual_funding_complete_2025-01-01_2025-10-01.parquet
    └── Funding rates (1,473 records, aligned to 8-hour intervals)
```

### Verification

After download completes, verify data quality:

```python
import pandas as pd

# Check OHLC data
ohlc = pd.read_parquet('crypto/data/historical/btc-perpetual_8H_2025-01-01_2025-10-01.parquet')
print(f"OHLC: {len(ohlc)} bars from {ohlc['timestamp'].min()} to {ohlc['timestamp'].max()}")

# Check funding data
funding = pd.read_parquet('crypto/data/historical/btc-perpetual_funding_complete_2025-01-01_2025-10-01.parquet')
print(f"Funding: {len(funding)} records from {funding['timestamp'].min()} to {funding['timestamp'].max()}")
```

### Common Issues

**No Tardis API key:**
- Free tier: Limited to first day of each month
- Solution: Get API key from https://tardis.dev or use Deribit-only mode for recent data

**Download fails or times out:**
- The script automatically retries failed downloads
- Missing days will be logged but don't block the overall download
- You can re-run for specific date ranges to fill gaps

**Data gaps:**
- Check logs for SSL errors or 404s (missing days in Tardis dataset)
- Re-run script with date range covering only missing days
- Script will merge new downloads with existing data

---

## Step 1: Explore Available Options

### Purpose
Search for liquid options at your target expiry and review candidates. The script uses historical data from Tardis.dev, querying the full trading day (00:00-23:59 UTC) to maximize discovery of traded options.

### Performance Benchmarks

| Search Type | Strikes Checked | Duration | Best For |
|-------------|----------------|----------|----------|
| **ATM (±5%)** | ~11 strikes | ~2 minutes | ✅ Recommended for quick option selection |
| **Wide (±30%)** | ~40 strikes | ~8 minutes | Comprehensive market scan |
| **Per-strike** | 1 strike | ~11 seconds | Network-bound (Tardis API) |

**Example**: Finding 1-week ATM calls takes approximately 2 minutes.

### Deribit Option Types

| Type | Expiry Schedule | Expiry Time | Notes |
|------|----------------|-------------|-------|
| **Daily** | Every day | 08:00 UTC | Listed ~48 hours before expiry |
| **Weekly** | Every Friday | 08:00 UTC | Most popular for short-term |
| **Monthly** | Last Friday of month | 08:00 UTC | Standard monthly expiries |
| **Quarterly** | Last Fri of Mar/Jun/Sep/Dec | 08:00 UTC | If falls on month-end, no separate monthly |

### Command

**Simplified format (recommended):**
```bash
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 \
    --expiry 2024-10-29 \
    --type call \
    --min-trades 10 \
    --data-source tardis \
    --output options_candidates.json
```

**With specific time (optional):**
```bash
python crypto/scripts/explore_options.py \
    --trade-date "2024-10-15 14:30" \
    --expiry "2024-10-29 08:00" \
    --type call \
    --min-trades 10 \
    --data-source tardis \
    --output options_candidates.json
```

### Parameters
- `--trade-date`: When you would sell the option
  - Simple: `2024-10-15` (defaults to 12:00 UTC for spot price)
  - Specific: `"2024-10-15 14:30"` (custom time)
- `--expiry`: Option expiry date
  - Simple: `2024-10-29` (defaults to 08:00 UTC, Deribit standard)
  - Specific: `"2024-10-29 08:00"`
- `--type`: `call` or `put`
- `--data-source`: Use `tardis` for historical data access (back to 2019)
- `--min-trades`: Minimum trades for liquidity filter (default: 10)
- `--moneyness-range`: Min/max moneyness (default: 0.9 1.1)
  - ATM search: `0.95 1.05` (~2 min, recommended)
  - Wide search: `0.7 1.3` (~8 min, comprehensive)
- `--testnet`: Use testnet data (for testing only)
- `--output`: JSON file to save results

### How It Works

The script uses a **full-day querying approach** to discover options:

1. **Spot Price**: Queries BTC-PERPETUAL trades in a ±5 minute window around `trade-date` for accurate spot price
2. **Strike Generation**: If Instruments API returns no data (expired options beyond ~1 month retention), automatically generates candidate strikes using Deribit's grid pattern (1K/2K/5K intervals)
3. **Full-Day Trade Queries**: For each strike, queries ALL trades from 00:00-23:59 UTC on the trade date to maximize finding historical option trades
4. **Liquidity Filter**: Filters by minimum trade count threshold
5. **Results**: Sorted by moneyness (ATM first)

### Output
```json
{
  "search_parameters": { ... },
  "options": [
    {
      "instrument_name": "BTC-29OCT24-50000-C",
      "strike": 50000,
      "option_type": "call",
      "initial_spot": 50250.0,
      "moneyness": 1.005,
      "premium_btc": 0.0624,
      "premium_usd": 3136.0,
      "implied_volatility": 0.85,
      "trade_count": 156,
      "days_to_expiry": 14
    },
    ...
  ]
}
```

### Human Decision Point
**Review the output table:**
```
================================================================================
AVAILABLE OPTIONS
================================================================================

Instrument              Strike  Moneyness  Premium (BTC)  Premium (USD)      IV  Trades
------------------------------------------------------------------------------------
BTC-29OCT24-50000-C   $50,000      1.005         0.0624      $3,136.00   85.0%     156
BTC-29OCT24-48000-C   $48,000      1.047         0.0892      $4,482.30   82.5%      89
BTC-29OCT24-52000-C   $52,000      0.966         0.0421      $2,115.50   87.2%     124
================================================================================
```

**Ask yourself:**
- Which strike offers best premium for the risk?
- Is liquidity sufficient (trade count)?
- Does IV seem reasonable?
- Is moneyness appropriate for your view?

**Select one option and note its `instrument_name`** (e.g., `BTC-29OCT24-50000-C`)

---

## Step 2: Train Model for Selected Option

### Purpose
Train a deep hedging model specifically for your selected option using realistic market parameters.

### Command
```bash
python crypto/scripts/train_for_option.py \
    --option-file options_candidates.json \
    --instrument BTC-29OCT24-50000-C \
    --epochs 100 \
    --paths 50000 \
    --output models/oct29_50k_call
```

### Parameters
- `--option-file`: JSON from Step 1
- `--instrument`: Instrument name you selected
- `--epochs`: Training epochs (default: 100)
- `--paths`: Training paths (more = better, slower; default: 50000)
- `--layers`: Hidden layers (default: 4)
- `--units`: Units per layer (default: 128)
- `--vol`: Override volatility (optional, uses IV from option if not specified)
- `--cost`: Transaction cost rate (default: 0.0006 = 0.06%)
- `--dt-hours`: Rebalancing frequency in hours (default: 8.0)
- `--output`: Directory to save model and results

### Output Files
```
models/oct29_50k_call/
├── model.pth                # Trained model checkpoint
├── training_results.json    # Training metrics
└── option_metadata.json     # Option parameters used
```

### Human Decision Point
**Review training results:**

```bash
cat models/oct29_50k_call/training_results.json
```

**Check:**
```json
{
  "training_history": [2340.12, 2250.45, ..., 1890.23],
  "test_metrics": {
    "deep_hedge": {
      "mean_pnl": -3200.50,
      "std_pnl": 450.25,
      "sharpe_ratio": -7.11
    },
    "bs_baseline": {
      "mean_pnl": -3450.80,
      "std_pnl": 550.12,
      "sharpe_ratio": -6.27
    }
  }
}
```

**Ask yourself:**
- Did loss improve during training?
- Is deep hedge better than BS baseline?
- Are Sharpe ratios reasonable?
- Do I trust this model?

**Decision: Proceed to backtest or retrain with different parameters?**

---

## Step 3: Backtest the Strategy

### Purpose
Test the trained model on historical data from trade date to expiry.

### Create Backtest Config

Create `backtest_config.yaml`:
```yaml
# Dates
start_date: "2024-10-15"  # Trade date
end_date: "2024-10-29"    # Expiry date

# Option parameters (from your selection)
strike: 50000
maturity_days: 14
call: true

# Bootstrap mode (NEW - see explanation below)
bootstrap_mode: "normalize_spot"  # RECOMMENDED
initial_spot: 50250.0             # From Step 1 option discovery

# Model
model_path: "models/oct29_50k_call/model.pth"

# Backtest parameters
n_bootstrap_paths: 1000      # More paths = more robust
transaction_cost: 0.0006     # 0.06%
dt_hours: 8.0                # 8-hour rebalancing

# Data
data_dir: "crypto/data/historical"
output_dir: "backtest_results/oct29_50k_call"
```

### Understanding Bootstrap Modes

**IMPORTANT**: The bootstrap mode determines how historical prices are used to create test scenarios, which critically affects backtest validity.

#### The Problem: Moneyness Consistency

When backtesting an option with strike $50K:
- January historical data: BTC was ~$42K → moneyness = 0.84 (deep OTM)
- October historical data: BTC was ~$108K → moneyness = 2.16 (deep ITM)

If we use a fixed strike with varying historical spots, each bootstrap path tests a **fundamentally different option**. The neural network sees completely different inputs (`log_moneyness`), and averaging P&L from OTM/ATM/ITM options together is **meaningless** for predicting performance of the specific option you plan to trade.

#### Solution: Two Bootstrap Modes

**1. `normalize_spot` (RECOMMENDED for trading decisions)**
- **What it does**: Rescales historical prices so all paths start at the same spot (preserving moneyness)
- **Why it works**: Multiplicative rescaling preserves returns and volatility (tested)
- **Result**: All bootstrap paths test the **exact same option characteristics** (e.g., all 98% OTM calls)
- **Configuration**:
  ```yaml
  bootstrap_mode: "normalize_spot"
  initial_spot: 50250.0  # From Step 1 (option discovery)
  # OR
  target_moneyness: 0.98  # Explicitly set if you don't have initial_spot
  ```

**2. `absolute_strike` (legacy, for research only)**
- **What it does**: Uses raw historical prices without rescaling
- **Result**: Each path tests a different option type (OTM/ATM/ITM mix)
- **Use case**: Historical analysis, understanding regime-dependent behavior
- **Configuration**:
  ```yaml
  bootstrap_mode: "absolute_strike"
  # No initial_spot needed
  ```

#### Which Mode Should I Use?

| Goal | Mode | Reason |
|------|------|--------|
| **Trading decision** (Should I sell this option?) | `normalize_spot` | Tests the specific option you'll trade |
| **Forward-looking P&L estimate** | `normalize_spot` | All paths match your option's characteristics |
| **Historical performance analysis** | `absolute_strike` | See how strategy performed across different regimes |
| **Research / regime studies** | `absolute_strike` | Understand behavior in different market conditions |

**For real trading: Always use `normalize_spot`** with the `initial_spot` from Step 1 option discovery.

#### How to Set initial_spot

The `initial_spot` is already in your `options_candidates.json` from Step 1:

```bash
# View the initial_spot for your selected option
cat options_candidates.json | grep -A 5 "BTC-29OCT24-50000-C" | grep initial_spot
```

This ensures your backtest uses the exact same spot price that was used to calculate the option's premium and moneyness.

### Command
```bash
python -m crypto.backtest.run \
    --config backtest_config.yaml \
    --seed 42
```

### Output Files
```
backtest_results/oct29_50k_call/
├── results.json          # P&L and metrics
├── backtest_report.md    # Detailed report
└── plots/
    ├── pnl_comparison.png
    ├── pnl_distribution.png
    └── positions.png
```

### Human Decision Point
**Review backtest report:**

```bash
cat backtest_results/oct29_50k_call/backtest_report.md
```

**Key metrics to check:**
```
Deep Hedge:
  Mean PnL: $-3,200 ± $450
  Sharpe Ratio: -7.11
  Max Drawdown: $-4,500
  Win Rate: 12%

Black-Scholes:
  Mean PnL: $-3,450 ± $550
  Sharpe Ratio: -6.27
```

**⚠️  IMPORTANT:** These are **hedging costs only** (negative because we're short the option).
You haven't added the premium yet!

**✅ ALWAYS CHECK: Bootstrap Variance**

Before trusting these statistics, verify that bootstrap variance is present:
- **Good**: `Std PnL: $450` (variance > 0)
- **Bad**: `Std PnL: $0.00` (zero variance = meaningless statistics)

If `Std PnL: $0.00` for both Deep Hedge and BS Baseline:
1. You didn't download enough historical data
2. All bootstrap paths are identical (see Step 0 data requirements)
3. **Solution**: Download more data and re-run backtest

**Expected behavior with proper data:**
- Deep Hedge `std`: Typically 20-50% of mean (reflects path diversity)
- BS Baseline `std`: Typically 10-30% of mean (less variance than deep hedge)
- Both should show `std > 0`, otherwise statistics are invalid

**Ask yourself:**
- Are hedging costs stable (low std)?
- Is deep hedge better than BS?
- Are drawdowns acceptable?
- **Do the statistics have variance (std > 0)?**

---

## Step 4: Calculate Expected Seller Returns

### Purpose
Combine premium received with hedging costs to get true expected P&L.

### Command
```bash
python crypto/scripts/calculate_seller_pnl.py \
    --option options_candidates.json \
    --instrument BTC-29OCT24-50000-C \
    --backtest backtest_results/oct29_50k_call/results.json \
    --output final_analysis.json
```

### Parameters
- `--option`: Option metadata from Step 1
- `--instrument`: Instrument name (optional if option file has only one)
- `--backtest`: Results from Step 3
- `--output`: Output file (optional)

### Output
```
================================================================================
SELLER P&L ANALYSIS
================================================================================

Option: BTC-29OCT24-50000-C
  Type: CALL
  Strike: $50,000
  Initial spot: $50,250
  Moneyness: 1.005
  Days to expiry: 14
  Premium: 0.0624 BTC ($3,136.00)

Premium Received (Income):
  0.0624 BTC
  $3,136.00 USD

Deep Hedge Strategy:
  Hedging P&L: $-3,200 ± $450
  Premium: $3,136
  ──────────────────────────────────────────────────
  Total Seller P&L: $-64 ± $450
  Sharpe Ratio: -0.142
  Win Rate: 47.5%

Black-Scholes Baseline:
  Hedging P&L: $-3,450 ± $550
  Premium: $3,136
  ──────────────────────────────────────────────────
  Total Seller P&L: $-314 ± $550
  Sharpe Ratio: -0.571
  Win Rate: 38.2%

Comparison (Deep Hedge vs Black-Scholes):
  P&L Improvement: $+250
  Risk Reduction: +18.2%
  Sharpe Improvement: +0.429

================================================================================
TRADING DECISION:
================================================================================
✗ UNPROFITABLE: Expected to lose $64 per option
  Hedging costs ($3,200) exceed premium ($3,136)
  Consider: Higher premium, shorter expiry, or different strike
================================================================================
```

### Final Decision
**Based on the analysis above, decide:**

**If Profitable (Total P&L > 0):**
- ✅ **Execute Trade**: Premium exceeds expected hedging costs
- Check: Sharpe ratio acceptable? Win rate sufficient?
- Position size: Start small, scale if profitable

**If Unprofitable (Total P&L < 0):**
- ❌ **Pass**: Premium doesn't cover hedging costs
- Consider: Different strike, expiry, or wait for better premium
- Or: Accept small loss if you have strong directional view

**Borderline (Small positive/negative):**
- Consider transaction costs in execution
- Review win rate and risk metrics
- Maybe reduce position size

---

## Complete Example Walkthrough

### Scenario: Trading BTC Call Options on Oct 15

```bash
# ============================================================================
# STEP 1: EXPLORE OPTIONS
# ============================================================================
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 \
    --expiry 2024-10-29 \
    --type call \
    --min-trades 10 \
    --data-source tardis \
    --output options_oct15.json

# Review output table...
# Decision: Select BTC-29OCT24-50000-C (premium: $3,136, IV: 85%)

# ============================================================================
# STEP 2: TRAIN MODEL
# ============================================================================
python crypto/scripts/train_for_option.py \
    --option-file options_oct15.json \
    --instrument BTC-29OCT24-50000-C \
    --epochs 100 \
    --paths 50000 \
    --output models/oct29_50k_call

# Review training_results.json...
# Decision: Model looks good, deep hedge beats BS baseline

# ============================================================================
# STEP 3: BACKTEST
# ============================================================================
# First, create backtest_oct15.yaml:
# - Use bootstrap_mode: "normalize_spot"
# - Set initial_spot: 50250.0 (from options_oct15.json)
# - See Step 3 for full config example

python -m crypto.backtest.run \
    --config backtest_oct15.yaml \
    --seed 42

# Review backtest_report.md and plots...
# Hedging P&L: -$3,200 ± $450 (stable, better than BS)
# Decision: Proceed to final analysis

# ============================================================================
# STEP 4: CALCULATE RETURNS
# ============================================================================
python crypto/scripts/calculate_seller_pnl.py \
    --option options_oct15.json \
    --instrument BTC-29OCT24-50000-C \
    --backtest backtest_results/oct29_50k_call/results.json \
    --output final_analysis.json

# Review output...
# Total P&L: -$64 ± $450
# Decision: PASS - premium slightly too low, wait for better opportunity
```

---

## Tips for Real Trading

### 1. Start with Exploration
Always explore multiple strikes and expiries before committing:
```bash
# Explore weekly options
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 \
    --expiry 2024-10-22 \
    --data-source tardis \
    --output weekly_options.json

# Explore monthly options
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 \
    --expiry 2024-11-15 \
    --data-source tardis \
    --output monthly_options.json
```

### 2. Reuse Trained Models
If you've already trained for a similar option, you can skip training:
```bash
# Backtest with existing model
python -m crypto.backtest.run \
    --config new_backtest.yaml \
    --model-path models/existing_model/model.pth
```

### 3. Sensitivity Analysis
Test different volatility assumptions:
```bash
# Train with higher volatility
python crypto/scripts/train_for_option.py \
    --option-file options.json \
    --instrument BTC-29OCT24-50000-C \
    --vol 0.95 \
    --output models/highvol

# Train with lower volatility
python crypto/scripts/train_for_option.py \
    --option-file options.json \
    --instrument BTC-29OCT24-50000-C \
    --vol 0.75 \
    --output models/lowvol
```

### 4. Batch Exploration
For research, explore many dates at once:
```bash
#!/bin/bash
for date in 2024-10-15 2024-10-22 2024-10-29; do
    python crypto/scripts/explore_options.py \
        --trade-date "$date" \
        --expiry 2024-11-15 \
        --data-source tardis \
        --output "options_${date}.json"
done
```

---

## When to Use Automated Pipeline

Use `realistic_backtest.py` (automated) when:
- Testing multiple scenarios in batch
- Research / parameter sweeps
- You don't need to review intermediate results

```bash
# Automated end-to-end (no human decisions)
python crypto/scripts/realistic_backtest.py \
    --config realistic_config.yaml
```

---

## Troubleshooting

### No options found
```
Error: No options found matching criteria
```
**Solutions:**
- Lower `--min-trades` threshold
- Widen `--moneyness-range`
- Try different expiry date
- Check if testnet has limited data

### Training fails
```
Error: CUDA out of memory
```
**Solutions:**
- Reduce `--paths` (e.g., 10000 instead of 50000)
- Reduce `--units` (e.g., 64 instead of 128)
- Use CPU: Remove CUDA requirements

### Backtest date mismatch
```
Error: No data found in date range
```
**Solutions:**
- Fetch data first: `python crypto/scripts/fetch_deribit_data.py`
- Check `data_dir` in config points to correct location
- Verify dates match option's lifetime

### All bootstrap paths are identical (variance = 0.00)
```
WARNING: Only 1 possible window!
All 100 paths will be IDENTICAL (zero variance in results).
```
**Symptom:** Backtest shows std = 0.00 for all metrics:
```
Deep Hedge: Mean PnL: $-2,671.19 ± $0.00
            Std PnL: $0.00
            Sharpe Ratio: -10886357.874
```

**Cause:** Insufficient historical data - downloaded exactly the backtest period with no extra data for random sampling.

**Example of the problem:**
- Downloaded: Oct 1-15 (14 days) = 42 records at 8H intervals
- Backtest needs: 14 days = 42 steps
- Bootstrap windows available: 42 - 42 + 1 = **1 window only!**
- Result: All 100 paths sample the same data → zero variance

**Solution:** Download MORE data than your backtest period!

**Formula:**
```bash
# For 14-day backtest with 100 unique windows:
extra_days = 100 * dt_hours / 24  # 100 * 8 / 24 = 33 days
total_days = backtest_days + extra_days  # 14 + 33 = 47 days

# Download command:
python crypto/scripts/fetch_deribit_data.py \
    --start 2024-09-01 \  # 33 days BEFORE backtest
    --end 2024-10-16      # 1 day after backtest
```

**Verification:** After fixing, re-run backtest and check for variance > 0:
```
Deep Hedge: Mean PnL: $-2,671.19 ± $719.49  ✓ Has variance
            Std PnL: $719.49                 ✓ Non-zero
```

---

## Summary Checklist

Before executing a trade, ensure:

- [ ] Explored multiple option candidates
- [ ] Selected option with good liquidity (trade count > 10)
- [ ] Trained model and reviewed training metrics
- [ ] Deep hedge beats Black-Scholes baseline
- [ ] Backtest shows stable hedging P&L
- [ ] Final seller P&L is positive (or acceptable loss)
- [ ] Win rate and Sharpe ratio are reasonable
- [ ] Understood all risks and position sizing

---

*Last updated: 2025-10-21*
*For automated workflows, see: realistic_backtest.py*
