# Crypto Scripts Reference

Complete reference for all executable scripts in the crypto module.

## Core Workflow Scripts

### explore_options.py
Query and explore available cryptocurrency options from Deribit/Tardis exchanges.

**What it does:**
- Fetches current spot price from perpetual market
- Lists available options filtered by maturity and moneyness
- Calculates implied volatility from option premiums
- Checks liquidity (open interest, bid/ask volume)
- Saves selected options to JSON for downstream use

**Usage:**
```bash
python -m crypto.scripts.explore_options --currency BTC --days-to-maturity 7-30 --moneyness 0.9-1.1
```

**Output:** JSON file with option metadata (strike, expiry, premium, IV, liquidity)

---

### train_for_option.py
Train a deep hedging model for a specific option from metadata file.

**What it does:**
- Loads option metadata from JSON (from explore_options.py)
- Creates training configuration with strike normalization
- Trains deep hedge neural network using historical volatility
- Saves model checkpoint and training config
- Optionally creates backtest config for next step

**Usage:**
```bash
python -m crypto.scripts.train_for_option --option-file options.json --instrument BTC-1FEB25-95000-C --epochs 100
```

**Output:** Model checkpoint (.pt), training config (.yaml), optional backtest config (.yaml)

---

### calculate_seller_pnl.py
Calculate expected P&L for option seller combining premium and hedging costs.

**What it does:**
- Loads backtest results (hedger P&L distribution)
- Loads option metadata (premium received)
- Computes seller P&L = premium + hedger_pnl
- Shows distribution statistics and risk metrics
- Helps decide whether to sell the option

**Usage:**
```bash
python -m crypto.scripts.calculate_seller_pnl --backtest-results results.json --option-metadata option.json
```

**Output:** Seller P&L statistics, expected profit, risk metrics

---

### fetch_deribit_data.py
Download historical market data from Deribit and Tardis APIs.

**What it does:**
- Downloads OHLC perpetual futures data (8H intervals)
- Downloads funding rate history
- Downloads spot trade data (via Tardis CSV API)
- Supports hybrid mode (Deribit API + Tardis CSV)
- Handles batching, resampling, timezone conversion

**Usage:**
```bash
python -m crypto.scripts.fetch_deribit_data --instrument BTC-PERPETUAL --start 2024-01-01 --end 2024-12-31 --data-type perpetual
```

**Output:** Parquet files with historical OHLC/funding/spot data

---

## Hyperparameter Tuning Scripts

### tune_for_option.py
Grid search over hyperparameters for deep hedging models.

**What it does:**
- Generates hyperparameter grid (n_layers, n_units, learning_rate, loss functions)
- Runs train_for_option.py for each combination
- Tracks all experiments with unique IDs
- Creates backtest configs for each trained model
- Saves tuning results and logs

**Usage:**
```bash
python -m crypto.scripts.tune_for_option --option-file options.json --instrument BTC-1FEB25-95000-C --grid-file hparam_grid.json
```

**Output:** Multiple model checkpoints, backtest configs, tuning log

---

### run_hparam_analysis.py
Master script to run complete hyperparameter analysis pipeline.

**What it does:**
- Orchestrates 4-stage analysis pipeline:
  1. Aggregate results (analyze_hparam_results.py)
  2. Compare and rank models (compare_hparam_models.py)
  3. Visualize distributions (visualize_hparam_results.py)
  4. Generate report (generate_hparam_report.py)
- Runs all stages sequentially
- Provides single entry point for analysis

**Usage:**
```bash
python -m crypto.scripts.run_hparam_analysis --results-dir tuning_results/ --output-dir analysis/
```

**Output:** Complete analysis with CSV rankings, plots, HTML report

---

### analyze_hparam_results.py
Aggregate training and backtest results from hyperparameter tuning runs.

**What it does:**
- Scans results directory for all training/backtest outputs
- Extracts key metrics (mean/std P&L, Sharpe, variance ratios)
- Filters valid hedgers (variance ratio < 1.0)
- Creates aggregated CSV and pickle files
- Handles missing/incomplete results gracefully

**Usage:**
```bash
python -m crypto.scripts.analyze_hparam_results --results-dir tuning_results/
```

**Output:** aggregated_results.csv, aggregated_results.pkl

---

### compare_hparam_models.py
Rank and compare models across different metrics.

**What it does:**
- Loads aggregated results from analyze_hparam_results.py
- Ranks models by mean P&L, Sharpe ratio, variance reduction
- Finds Pareto-optimal models (best tradeoffs)
- Analyzes which architectures perform best
- Generates ranking CSVs for each metric

**Usage:**
```bash
python -m crypto.scripts.compare_hparam_models --aggregated-results aggregated_results.pkl
```

**Output:** rankings_mean_pnl.csv, rankings_sharpe.csv, architecture_analysis.csv

---

### visualize_hparam_results.py
Create visualizations for hyperparameter tuning results.

**What it does:**
- Generates ~10 different plot types:
  - P&L distributions, heatmaps, correlation matrices
  - Learning rate effects, loss function comparisons
  - Architecture performance, Pareto frontiers
- Saves all plots to output directory
- Creates publication-ready figures

**Usage:**
```bash
python -m crypto.scripts.visualize_hparam_results --aggregated-results aggregated_results.pkl --output-dir plots/
```

**Output:** Multiple PNG/PDF plots in output directory

---

### generate_hparam_report.py
Generate comprehensive Markdown/HTML report from tuning results.

**What it does:**
- Combines analysis from all previous scripts
- Creates executive summary with key findings
- Shows top models, architecture insights, recommendations
- Generates interactive HTML with embedded plots
- Provides actionable insights for model selection

**Usage:**
```bash
python -m crypto.scripts.generate_hparam_report --aggregated-results aggregated_results.pkl --output report.md
```

**Output:** Markdown report, optional HTML version

---

## Configuration & Utilities

### create_backtest_config.py
Generate backtest configuration YAML from training results and option metadata.

**What it does:**
- Loads trained model checkpoint
- Extracts option metadata (strike, maturity, premium)
- Creates backtest config with correct paths and parameters
- Ensures compatibility between training and backtest settings
- Used by tune_for_option.py and train_for_option.py

**Usage:**
```bash
python -m crypto.scripts.create_backtest_config --model-path model.pt --option-metadata option.json --output backtest.yaml
```

**Output:** backtest.yaml ready for crypto.backtest.run

---

### compute_market_params.py
Estimate market parameters (volatility, drift) from recent historical data.

**What it does:**
- Fetches recent BTC perpetual data from Deribit
- Computes realized volatility (rolling window)
- Estimates drift from price trends
- Provides parameters for training configuration
- Useful for calibrating models to current market conditions

**Usage:**
```bash
python -m crypto.scripts.compute_market_params --days 30 --window-hours 24
```

**Output:** Estimated volatility and drift parameters

---

## Testing & Diagnostics

### test_deribit_client.py
Test Deribit API client connectivity and functionality.

**What it does:**
- Tests all DeribitClient methods (instruments, ticker, order book, trades)
- Verifies API authentication (testnet and mainnet)
- Checks data quality and response times
- Useful for debugging API issues
- Can run on testnet without affecting real account

**Usage:**
```bash
python -m crypto.scripts.test_deribit_client --testnet
```

**Output:** Test results showing API connectivity status

---

### verify_tardis.py
Comprehensive verification of Tardis API connectivity and data quality.

**What it does:**
- Tests 5 different Tardis data endpoints:
  - Spot trades CSV downloads
  - Index price CSV downloads
  - Perpetual OHLC CSV downloads
  - Funding rate CSV downloads
  - API authentication
- Checks data completeness and consistency
- Verifies timestamp handling and decompression
- Essential for setup and troubleshooting

**Usage:**
```bash
python -m crypto.scripts.verify_tardis --api-key YOUR_KEY
```

**Output:** Detailed test report with data samples

---

### verify_checkpoint.py
Verify model checkpoint has correct strike normalization and architecture.

**What it does:**
- Loads model checkpoint file
- Checks normalized strike matches option metadata
- Verifies model architecture (layers, units)
- Validates training configuration consistency
- Helps debug training/backtest mismatches

**Usage:**
```bash
python -m crypto.scripts.verify_checkpoint --checkpoint model.pt --option-metadata option.json
```

**Output:** Checkpoint validation report

---

### diagnose_gpu.py
Comprehensive GPU diagnostics for deep learning setup.

**What it does:**
- Checks CUDA availability and version
- Tests GPU memory allocation and throughput
- Benchmarks pfhedge training performance on GPU
- Compares CPU vs GPU training speed
- Identifies GPU configuration issues

**Usage:**
```bash
python -m crypto.scripts.diagnose_gpu
```

**Output:** GPU diagnostic report with performance benchmarks

---

### profile_training.py
Profile training performance to identify bottlenecks.

**What it does:**
- Runs training with profiling enabled
- Measures time spent in each operation
- Identifies slow components (data loading, forward pass, backward pass)
- Suggests optimization opportunities
- Useful for improving training speed

**Usage:**
```bash
python -m crypto.scripts.profile_training --config training_config.yaml
```

**Output:** Performance profile with timing breakdown

---

### debug_pnl.py
Compare different P&L calculation methods for debugging.

**What it does:**
- Computes P&L using multiple methods (manual, pfhedge native)
- Compares Black-Scholes delta hedging vs deep hedge
- Shows step-by-step P&L breakdown
- Helps debug unexpected P&L results
- Validates calculation correctness

**Usage:**
```bash
python -m crypto.scripts.debug_pnl
```

**Output:** P&L comparison showing calculation differences

---

## Quick Reference

**Typical workflow:**
1. `explore_options.py` - Find options
2. `train_for_option.py` - Train model
3. `python -m crypto.backtest --config backtest.yaml` - Backtest
4. `calculate_seller_pnl.py` - Calculate returns

**For hyperparameter tuning:**
1. `explore_options.py` - Find option
2. `tune_for_option.py` - Grid search
3. `run_hparam_analysis.py` - Analyze results
4. Pick best model and backtest

**For debugging:**
- Data issues: `verify_tardis.py`, `test_deribit_client.py`
- GPU issues: `diagnose_gpu.py`
- Training slow: `profile_training.py`
- Wrong P&L: `debug_pnl.py`, `verify_checkpoint.py`
