# Deep Hedging Quick Instructions

Complete guide to running the deep hedging workflow from data download to final P&L analysis.

---

## Quick Start (4 Commands)

```bash
# 1. Download historical data
python crypto/scripts/fetch_deribit_data.py \
    --start 2024-09-01 --end 2024-10-29 \
    --output-dir crypto/data/historical

# 2. Find available options
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 --expiry 2024-10-29 \
    --type call --output options.json

# 3. Train hedging model
python crypto/scripts/train_for_option.py \
    --option-file options.json \
    --instrument BTC-29OCT24-50000-C \
    --output models/my_model

# 4. Run backtest + calculate returns
python -m crypto.backtest.run --config backtest.yaml
python crypto/scripts/calculate_seller_pnl.py \
    --option options.json \
    --instrument BTC-29OCT24-50000-C \
    --backtest backtest_results/results.json
```

---

## Step 0: Download Historical Data

### Why This Step?
Backtesting requires historical BTC-PERPETUAL prices and funding rates. You must download MORE data than your backtest period to get meaningful bootstrap statistics.

### Data Requirements

**Formula**: `download_days = backtest_days + 33 extra days`

| Backtest Period | Download Period | Why |
|-----------------|-----------------|-----|
| Oct 1-15 (14 days) | Sep 1 - Oct 16 (47 days) | Need 100+ unique bootstrap windows |
| Oct 1-30 (30 days) | Sep 1 - Oct 31 (63 days) | Extra data for path diversity |

**What happens if you don't?**
- Download exactly 14 days → Only 1 bootstrap window → All paths identical → Std = 0.00 (meaningless)
- Download 47 days → 100+ bootstrap windows → Valid statistics with variance > 0

### How to Run

**Basic command** (downloads OHLC + funding rates):
```bash
python crypto/scripts/fetch_deribit_data.py \
    --start 2024-09-01 \
    --end 2024-10-29 \
    --output-dir crypto/data/historical
```

**With Tardis API key** (for historical data >30 days):
```bash
export TARDIS_API_KEY=your_key_here

python crypto/scripts/fetch_deribit_data.py \
    --data-source tardis \
    --start 2024-01-01 \
    --end 2024-10-29 \
    --output-dir crypto/data/historical \
    --tardis-api-key $TARDIS_API_KEY
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--start` | Required | Start date (YYYY-MM-DD) |
| `--end` | Required | End date (YYYY-MM-DD) |
| `--output-dir` | `crypto/data/historical` | Where to save data |
| `--data-source` | `deribit` | `deribit` (recent 30 days) or `tardis` (back to 2019) |
| `--instrument` | `BTC-PERPETUAL` | Perpetual contract name |
| `--frequency` | `8H` | Resampling frequency (8-hour bars) |
| `--tardis-api-key` | `$TARDIS_API_KEY` | API key for Tardis (required for historical data) |

### Output Files

```
crypto/data/historical/
├── btc-perpetual_8H_2024-09-01_2024-10-29.parquet      # OHLC prices
└── btc-perpetual_funding_complete_2024-09-01_2024-10-29.parquet  # Funding rates
```

---

## Step 1: Explore Available Options

### Why This Step?
Find liquid, tradeable options at your target date and expiry. Review premiums and strike prices before selecting.

### How to Run

**Basic command** (ATM options, ~2 minutes):
```bash
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 \
    --expiry 2024-10-29 \
    --type call \
    --min-trades 10 \
    --output options.json
```

**Wide search** (all strikes, ~8 minutes):
```bash
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 \
    --expiry 2024-10-29 \
    --type call \
    --min-trades 5 \
    --moneyness-range 0.7 1.3 \
    --data-source tardis \
    --output options.json
```

**With specific time**:
```bash
python crypto/scripts/explore_options.py \
    --trade-date "2024-10-15 14:30" \
    --expiry "2024-10-29 08:00" \
    --type put \
    --output options.json
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trade-date` | Required | When you sell the option (YYYY-MM-DD or "YYYY-MM-DD HH:MM") |
| `--expiry` | Required | Option expiry date (YYYY-MM-DD, defaults to 08:00 UTC) |
| `--type` | Required | `call` or `put` |
| `--min-trades` | `10` | Minimum trades for liquidity filter |
| `--moneyness-range` | `0.9 1.1` | Search range (0.95 1.05 for ATM only) |
| `--data-source` | `tardis` | Use `tardis` for historical data |
| `--output` | stdout | JSON file to save results |

### Output

**Terminal output**:
```
================================================================================
AVAILABLE OPTIONS
================================================================================

Instrument              Strike  Moneyness  Premium (BTC)  Premium (USD)      IV  Trades
------------------------------------------------------------------------------------
BTC-29OCT24-50000-C   $50,000      1.005         0.0624      $3,136.00   85.0%     156
BTC-29OCT24-48000-C   $48,000      1.047         0.0892      $4,482.30   82.5%      89
BTC-29OCT24-52000-C   $52,000      0.966         0.0421      $2,115.50   87.2%     124
```

**JSON file** (`options.json`):
```json
{
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
    }
  ]
}
```

### What to Do Next

1. Review the output table
2. **Select one option** by noting its `instrument_name` (e.g., `BTC-29OCT24-50000-C`)
3. Consider:
   - Premium amount (is it worth the risk?)
   - Liquidity (trade count > 10?)
   - Implied volatility (reasonable?)
   - Moneyness (matches your view?)

---

## Step 2: Train Hedging Model

### Why This Step?
Train a deep learning model to learn optimal hedging strategies for your selected option, minimizing risk-adjusted P&L.

### How to Run

**Basic command** (recommended settings):
```bash
python crypto/scripts/train_for_option.py \
    --option-file options.json \
    --instrument BTC-29OCT24-50000-C \
    --output models/my_model
```

**Production settings** (more paths, more epochs):
```bash
python crypto/scripts/train_for_option.py \
    --option-file options.json \
    --instrument BTC-29OCT24-50000-C \
    --epochs 100 \
    --paths 500000 \
    --layers 4 \
    --units 128 \
    --risk-measure expected_shortfall \
    --risk-param 0.85 \
    --output models/production_model
```

**GPU training** (faster):
```bash
python crypto/scripts/train_for_option.py \
    --option-file options.json \
    --instrument BTC-29OCT24-50000-C \
    --device cuda \
    --output models/gpu_model
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--option-file` | Required | JSON from Step 1 (explore_options.py) |
| `--instrument` | Required | Instrument name (e.g., BTC-29OCT24-50000-C) |
| `--output` | Required | Directory to save model and results |
| `--epochs` | `100` | Training epochs (more = better, slower) |
| `--paths` | `50000` | Training paths (more = smoother, more memory) |
| `--layers` | `4` | Number of hidden layers in MLP |
| `--units` | `128` | Units per layer |
| `--risk-measure` | `expected_shortfall` | `expected_shortfall`, `entropic`, `cvar`, `variance` |
| `--risk-param` | `0.9` | Risk parameter (CVaR alpha or entropic a) |
| `--cost` | `0.0006` | Transaction cost rate (0.06%) |
| `--dt-hours` | `8.0` | Rebalancing frequency (8 hours = 3x daily) |
| `--vol` | from option IV | Override volatility (optional) |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` |
| `--seed` | `42` | Random seed for reproducibility |

### Risk Measures Explained

| Risk Measure | Parameter | Meaning | Typical Value |
|--------------|-----------|---------|---------------|
| `expected_shortfall` | `p` (alpha) | Average of worst p% scenarios | 0.85 (worst 15%) |
| `entropic` | `a` | Exponential utility, risk aversion | 0.1 (very conservative) |
| `cvar` | `lam` | Quadratic CVaR | 10.0 |
| `variance` | - | Minimize P&L variance | - |

**Recommendation**: Use `expected_shortfall` with `--risk-param 0.85` for interpretable results.

### Output Files

```
models/my_model/
├── model.pth                # Trained model checkpoint
├── training_results.json    # Training metrics and history
└── option_metadata.json     # Option parameters used
```

### What to Check

Review `training_results.json`:
```bash
cat models/my_model/training_results.json
```

Look for:
- **Training loss decreasing**: Model is learning
- **Deep hedge better than BS**: Lower mean P&L, better Sharpe ratio
- **Stable Sharpe ratio**: Around -5 to -10 is typical for hedging cost

---

## Step 3: Backtest the Strategy

### Why This Step?
Test the trained model on historical data to estimate realistic hedging costs and strategy performance.

### Create Backtest Config

Create `backtest.yaml`:
```yaml
# Dates
start_date: "2024-10-15"  # Trade date from Step 1
end_date: "2024-10-29"    # Expiry date from Step 1

# Option parameters (from Step 1 JSON)
strike: 50000
maturity_days: 14
call: true

# Bootstrap mode (CRITICAL - see explanation below)
bootstrap_mode: "normalize_spot"  # RECOMMENDED for trading decisions
initial_spot: 50250.0             # From Step 1 (options.json -> initial_spot)

# Model
model_path: "models/my_model/model.pth"

# Backtest parameters
n_bootstrap_paths: 1000      # More = more robust statistics
transaction_cost: 0.0006     # 0.06% (should match training)
dt_hours: 8.0                # 8-hour rebalancing (should match training)

# Data
data_dir: "crypto/data/historical"
output_dir: "backtest_results"
```

### Bootstrap Modes Explained

| Mode | When to Use | Effect |
|------|-------------|--------|
| `normalize_spot` | **Trading decisions** (recommended) | Rescales historical prices to preserve moneyness - all paths test the SAME option |
| `absolute_strike` | Research/historical analysis | Uses raw historical prices - each path tests different option types (OTM/ATM/ITM mix) |

**For real trading**: Always use `normalize_spot` with `initial_spot` from Step 1!

### How to Run

```bash
python -m crypto.backtest.run \
    --config backtest.yaml \
    --seed 42
```

**Override config parameters**:
```bash
python -m crypto.backtest.run \
    --config backtest.yaml \
    --seed 42 \
    --n-bootstrap-paths 2000 \
    --output-dir backtest_results/run2
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | Required | Path to YAML config file |
| `--seed` | Recommended | Random seed for reproducibility |
| `--n-bootstrap-paths` | from config | Override number of bootstrap paths |
| `--output-dir` | from config | Override output directory |
| `--diagnostics` | - | Enable MLP diagnostics during hedging |
| `--dry-run` | - | Print config without running |
| `--no-report` | - | Skip generating markdown report |

### Output Files

```
backtest_results/
├── results.json           # P&L metrics and statistics
├── backtest_report.md     # Detailed report with analysis
└── plots/
    ├── pnl_comparison.png      # Deep hedge vs BS baseline
    ├── pnl_distribution.png    # P&L histogram
    └── hedging_positions.png   # Position sizes over time
```

### What to Check

**Read the report**:
```bash
cat backtest_results/backtest_report.md
```

**Key metrics**:
```
Deep Hedge Strategy:
  Mean P&L: $-3,200 ± $450
  Std P&L: $450
  Sharpe Ratio: -7.11
  Max Drawdown: $-4,500
  Win Rate: 12%

Black-Scholes Baseline:
  Mean P&L: $-3,450 ± $550
  Std P&L: $550
  Sharpe Ratio: -6.27
```

**CRITICAL**: Check that `Std P&L > 0` for both strategies!
- If `Std P&L: $0.00` → You didn't download enough data (see Step 0)
- Expected: Std should be 20-50% of mean

**Understanding the numbers**:
- **Negative P&L** = Hedging cost (you haven't added premium yet!)
- **Lower absolute value** = Better (less expensive to hedge)
- **Deep hedge < BS** = Your model learned something useful

### Debugging Model Behavior (Optional)

If your model produces unexpectedly flat or constant hedge ratios, use diagnostics to inspect what's happening inside the neural network:

```bash
python -m crypto.backtest.run \
    --config backtest.yaml \
    --seed 42 \
    --diagnostics  # Add this flag
```

**Or in YAML config**:
```yaml
enable_diagnostics: true
```

**What diagnostics show**:
```
======================================================================
BACKTEST DIAGNOSTICS
======================================================================

📤 OUTPUT STATISTICS (Hedge Ratios):
   Mean: 0.167 ± 0.0002
   Std:  0.0003              ← Very low = flat hedging!
   Range: [0.166, 0.168]     ← Tiny range = model not adapting
```

**Interpreting results**:
- **Std < 0.001**: Model outputs nearly constant hedges (flat hedging issue)
- **Std > 0.01**: Model dynamically adjusts hedges (healthy)
- **Range < 0.002**: Model ignores market conditions
- **Range > 0.05**: Model responds to inputs

**Common issues detected**:
- **Flat hedging**: Output std < 0.001 → Check if `prev_hedge` feature dominates
- **No variation**: Model may need retraining without `prev_hedge` feature
- **Feedback loop**: Model copies previous hedge instead of computing optimal hedge

For detailed diagnostics documentation, see: `BACKTEST_DIAGNOSTICS.md`

---

## Step 4: Calculate Final Seller Returns

### Why This Step?
Combine the premium you receive from selling the option with the hedging cost to get your true expected profit/loss.

### How to Run

```bash
python crypto/scripts/calculate_seller_pnl.py \
    --option options.json \
    --instrument BTC-29OCT24-50000-C \
    --backtest backtest_results/results.json \
    --output final_analysis.json
```

If `options.json` has only one option, you can omit `--instrument`:
```bash
python crypto/scripts/calculate_seller_pnl.py \
    --option options.json \
    --backtest backtest_results/results.json
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--option` | Required | Option metadata from Step 1 |
| `--backtest` | Required | Backtest results from Step 3 |
| `--instrument` | Auto if single | Instrument name (required if multiple options) |
| `--output` | stdout | JSON file to save analysis |

### Output

**Terminal output**:
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

### Decision Rules

**If Total Seller P&L > 0**:
- ✅ **PROFITABLE** - Premium exceeds hedging costs
- Consider executing the trade
- Check: Win rate > 40%? Sharpe ratio acceptable?

**If Total Seller P&L < 0**:
- ❌ **UNPROFITABLE** - Premium doesn't cover hedging costs
- Pass on this trade
- Try: Different strike, expiry, or wait for better premium

**Borderline (±$100)**:
- Consider transaction costs in real execution
- Maybe reduce position size
- Review risk metrics carefully

---

## Complete Example Walkthrough

### Scenario: Selling BTC Call Option on Oct 15

```bash
# ============================================================================
# STEP 0: DOWNLOAD DATA (Download 47 days for 14-day backtest)
# ============================================================================
python crypto/scripts/fetch_deribit_data.py \
    --start 2024-09-01 \
    --end 2024-10-29 \
    --output-dir crypto/data/historical

# Output:
# crypto/data/historical/
# ├── btc-perpetual_8H_2024-09-01_2024-10-29.parquet
# └── btc-perpetual_funding_complete_2024-09-01_2024-10-29.parquet

# ============================================================================
# STEP 1: EXPLORE OPTIONS
# ============================================================================
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 \
    --expiry 2024-10-29 \
    --type call \
    --min-trades 10 \
    --output options_oct15.json

# Review output table, select: BTC-29OCT24-50000-C (premium: $3,136, IV: 85%)

# ============================================================================
# STEP 2: TRAIN MODEL
# ============================================================================
python crypto/scripts/train_for_option.py \
    --option-file options_oct15.json \
    --instrument BTC-29OCT24-50000-C \
    --epochs 100 \
    --paths 50000 \
    --output models/oct29_50k_call

# Review: models/oct29_50k_call/training_results.json
# Check: Deep hedge beats BS baseline ✓

# ============================================================================
# STEP 3: BACKTEST
# ============================================================================
# First, create backtest_oct15.yaml with:
# - start_date: "2024-10-15"
# - end_date: "2024-10-29"
# - bootstrap_mode: "normalize_spot"
# - initial_spot: 50250.0  # From options_oct15.json
# - model_path: "models/oct29_50k_call/model.pth"

python -m crypto.backtest.run \
    --config backtest_oct15.yaml \
    --seed 42

# Review: backtest_results/backtest_report.md
# Hedging P&L: -$3,200 ± $450 (stable, better than BS)

# ============================================================================
# STEP 4: CALCULATE RETURNS
# ============================================================================
python crypto/scripts/calculate_seller_pnl.py \
    --option options_oct15.json \
    --instrument BTC-29OCT24-50000-C \
    --backtest backtest_results/results.json \
    --output final_analysis.json

# Result: Total P&L: -$64 ± $450
# Decision: PASS - Premium slightly too low, wait for better opportunity
```

---

## Common Parameter Reference

### Transaction Costs

| Value | Meaning | When to Use |
|-------|---------|-------------|
| `0.0006` | 0.06% (Deribit maker) | **Recommended default** |
| `0.0010` | 0.10% (Deribit taker) | If you expect to take liquidity |
| `0.0000` | Zero cost | Testing only (unrealistic) |

### Rebalancing Frequency

| `--dt-hours` | Meaning | Trade-off |
|--------------|---------|-----------|
| `1.0` | Hourly | More hedging accuracy, higher transaction costs |
| `8.0` | 3x daily | **Recommended** (aligned with funding, balanced costs) |
| `24.0` | Daily | Lower costs, less accurate hedging |

### Training Paths

| `--paths` | Memory | Training Time | Quality |
|-----------|--------|---------------|---------|
| `10000` | Low | Fast (~30s/epoch) | Quick testing |
| `50000` | Medium | Medium (~2min/epoch) | **Recommended** |
| `500000` | High | Slow (~15min/epoch) | Production quality |

### Diagnostics

Use `--diagnostics` flag in training or backtesting to monitor model behavior:

| Use Case | When | What it Shows |
|----------|------|---------------|
| Training | `train_for_option.py --diagnostics` | Learning progress, gradient health, feature usage |
| Backtesting | `backtest.run --diagnostics` | Model behavior during hedging, flat hedging detection |

### Bootstrap Paths

| `--n-bootstrap-paths` | Quality | Runtime |
|-----------------------|---------|---------|
| `100` | Quick test | Fast (~10s) |
| `1000` | **Recommended** | Medium (~1min) |
| `10000` | High confidence | Slow (~10min) |

---

## Troubleshooting

### No options found
```
Error: No options found matching criteria
```
**Solutions**:
- Lower `--min-trades` to 5 or 1
- Widen `--moneyness-range` to `0.7 1.3`
- Try different expiry date

### Training fails (OOM)
```
Error: CUDA out of memory
```
**Solutions**:
- Reduce `--paths` to 10000
- Reduce `--units` to 64
- Use `--device cpu`

### Backtest: No data found
```
Error: No data found in date range
```
**Solutions**:
- Run Step 0 first (download data)
- Check `data_dir` in config matches download location
- Verify dates: backtest dates must be within downloaded range

### Backtest: Zero variance (std = 0.00)
```
WARNING: Only 1 possible window!
All bootstrap paths will be IDENTICAL
Deep Hedge: Mean PnL: $-2,671 ± $0.00
```
**Problem**: You didn't download enough data (see Step 0)

**Solution**: Download more data!
```bash
# For 14-day backtest, download 47 days:
python crypto/scripts/fetch_deribit_data.py \
    --start 2024-09-01 \  # 33 days BEFORE backtest start
    --end 2024-10-16      # 1 day after backtest end
```

**Formula**: `extra_days = 100 * dt_hours / 24` (for 100 unique windows)
- 14-day backtest with dt=8H → Need 47 days total

---

## Tips for Real Trading

### 1. Start with Quick Exploration
```bash
# ATM only (fast ~2min)
python crypto/scripts/explore_options.py \
    --trade-date 2024-10-15 --expiry 2024-10-22 \
    --moneyness-range 0.95 1.05 \
    --output weekly_atm.json
```

### 2. Reuse Trained Models
If you've trained for similar options, skip training:
```bash
# Use existing model for backtest
python -m crypto.backtest.run \
    --config new_backtest.yaml \
    # Model path already in config
```

### 3. Sensitivity Analysis
Test different volatility assumptions:
```bash
# High vol scenario
python crypto/scripts/train_for_option.py \
    --option-file options.json --instrument BTC-29OCT24-50000-C \
    --vol 0.95 --output models/highvol

# Low vol scenario
python crypto/scripts/train_for_option.py \
    --option-file options.json --instrument BTC-29OCT24-50000-C \
    --vol 0.75 --output models/lowvol
```

### 4. Batch Processing
For research, explore many dates:
```bash
#!/bin/bash
for date in 2024-10-{15,22,29}; do
    python crypto/scripts/explore_options.py \
        --trade-date "$date" --expiry 2024-11-15 \
        --output "options_${date}.json"
done
```

---

## Quick Reference

### File Locations

| File | Location | Purpose |
|------|----------|---------|
| Historical data | `crypto/data/historical/*.parquet` | OHLC + funding rates |
| Options list | `options.json` | Available options from Step 1 |
| Trained model | `models/my_model/model.pth` | Neural network weights |
| Backtest config | `backtest.yaml` | Backtest parameters |
| Backtest results | `backtest_results/results.json` | P&L metrics |
| Final analysis | `final_analysis.json` | Seller P&L summary |

### Deribit Option Expiries

| Type | Schedule | Time |
|------|----------|------|
| Daily | Every day | 08:00 UTC |
| Weekly | Every Friday | 08:00 UTC |
| Monthly | Last Friday of month | 08:00 UTC |
| Quarterly | Last Fri of Mar/Jun/Sep/Dec | 08:00 UTC |

---

*For detailed explanations and background, see: [STEP_BY_STEP_WORKFLOW.md](STEP_BY_STEP_WORKFLOW.md)*

*Last updated: 2025-10-27*
