# Step-by-Step Interactive Trading Workflow

Complete guide for interactive option trading with human decision points.

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [FAQ & Glossary](#faq--glossary)
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
# 1. Explore options
python crypto/scripts/explore_options.py \
    --trade-date "2024-10-15 12:00" \
    --expiry 2024-10-29 \
    --type call \
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

### Supporting Scripts

- **`fetch_deribit_data.py`** - Download historical data from Deribit (required before backtesting)
  ```bash
  python crypto/scripts/fetch_deribit_data.py \
      --start 2024-10-01 --end 2024-10-31 \
      --output-dir crypto/data/historical
  ```

- **`realistic_backtest.py`** - Automated end-to-end pipeline (for batch jobs only)
  ```bash
  python crypto/scripts/realistic_backtest.py \
      --config crypto/configs/realistic_backtest_example.yaml
  ```

---

## FAQ & Glossary

### Common Questions

**Q: Do I use testnet or mainnet for real trading?**
A: **Mainnet** (real data) by default. Only use `--testnet` flag for testing the scripts. Testnet has fake/limited data.

**Q: What currency are P&L calculations in?**
A: **USD**. Premiums are quoted in BTC but converted to USD for P&L reporting. Both BTC and USD values are saved.

**Q: What does "±" mean in results like "$-3,200 ± $450"?**
A: **Standard deviation** (risk/uncertainty). The mean is $-3,200 and ±$450 is the variation:
- 68% of scenarios: between $-3,650 and $-2,750
- 95% of scenarios: between $-4,100 and $-2,300
- Smaller ± = more consistent/predictable

**Q: Where does the data come from?**
A: **Deribit Public API** (no authentication needed for historical data):

| Data Type | API Method | What We Get |
|-----------|------------|-------------|
| Available options | `get_instruments` | List of all BTC options with strikes, expiries |
| Option premiums | `get_last_trades_by_instrument` | Actual executed trades (price, size, time) |
| Perpetual prices | `get_tradingview_chart_data` | OHLC bars at 8-hour intervals |
| Funding rates | `get_funding_rate_history` | 8-hour funding payments |

All data comes from **mainnet** by default (real market data). Use `--testnet` only for testing scripts.

**Q: How is the spot price determined for BTC/USD conversion?**
A: The spot price is **always taken at the trade date** (when you would execute the option trade). This ensures all conversions use the correct market price at the time of trading:

**Data Flow:**
1. **Step 1 (explore_options.py)**: Fetches BTC-PERPETUAL trades at `--trade-date`, uses median price as `initial_spot`
2. **Premium conversion**: `premium_usd = premium_btc * initial_spot`
3. **All subsequent steps**: Use the same `initial_spot` from option metadata

**Why this matters:**
- Premiums are quoted in BTC on Deribit (e.g., 0.0624 BTC)
- Converting to USD requires the BTC price at that moment
- Using the wrong spot price would distort P&L calculations
- All scripts consistently use `initial_spot` from the trade date

**Example:**
```
Trade date: 2024-10-15 12:00 UTC
BTC spot at that time: $50,250 (from perpetual trades)
Premium: 0.0624 BTC
Converted: 0.0624 × $50,250 = $3,136 USD
```

**Verification:** All BTC/USD conversions across the codebase use `initial_spot` consistently:
- `explore_options.py` line 90, 150
- `calculate_seller_pnl.py` line 118
- `realistic_backtest.py` line 325
- `select_option.py` line 322, 385
- `black_scholes.py` line 240

### Key Terms

**Trade Date** (`--trade-date`): When you would execute the option trade (formerly called "trade date")

**Trades** (in option table): Number of actual executed buy/sell transactions for that option. Higher = more liquid. Recommended minimum: 10.

**Implied Volatility (IV)**: **We calculate this**, not from exchange. We take the actual premium from trades and invert Black-Scholes formula to get volatility. This is the market's expectation of future price movement.

**Premium**: Price to buy/sell the option
- Quoted in **BTC** on Deribit (e.g., 0.0624 BTC)
- Converted to **USD** for analysis (e.g., $3,136)

**Hedging P&L**: Cost of maintaining the hedge position
- Usually **negative** for sellers (spending money to hedge)
- Includes transaction costs and funding fees

**Total Seller P&L**: The complete picture
```
Total P&L = Premium Received + Hedging P&L
          = (positive income) + (negative cost)
```

**Moneyness**: Ratio of spot price to strike price
- 1.0 = At-the-money (ATM)
- > 1.0 = In-the-money (ITM) for calls
- < 1.0 = Out-of-the-money (OTM) for calls

**Bootstrap Paths**: Multiple simulated price paths from historical data
- Used to test how strategy performs across different scenarios
- More paths = more robust results (but slower)

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

## Step 1: Explore Available Options

### Purpose
Search for liquid options at your target expiry and review candidates.

### Command
```bash
python crypto/scripts/explore_options.py \
    --trade-date "2024-10-15 12:00" \
    --expiry 2024-10-29 \
    --type call \
    --min-trades 10 \
    --output options_candidates.json
```

### Parameters
- `--trade-date`: When you would sell the option (YYYY-MM-DD HH:MM)
- `--expiry`: Option expiry date (YYYY-MM-DD)
- `--type`: `call` or `put`
- `--min-trades`: Minimum trades for liquidity filter (default: 10)
- `--moneyness-range`: Min/max moneyness (default: 0.9 1.1)
- `--testnet`: Use testnet data (for testing)
- `--output`: JSON file to save results

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

**Ask yourself:**
- Are hedging costs stable (low std)?
- Is deep hedge better than BS?
- Are drawdowns acceptable?

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
    --trade-date "2024-10-15 12:00" \
    --expiry 2024-10-29 \
    --type call \
    --min-trades 10 \
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
# First, create backtest_config.yaml (see Step 3 above)

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
    --trade-date "2024-10-15 12:00" \
    --expiry 2024-10-22 \
    --output weekly_options.json

# Explore monthly options
python crypto/scripts/explore_options.py \
    --trade-date "2024-10-15 12:00" \
    --expiry 2024-11-15 \
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
        --trade-date "$date 12:00" \
        --expiry 2024-11-15 \
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
