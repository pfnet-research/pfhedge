# Deep Hedging Backtest Framework - User Guide

Complete guide for running deep hedging backtests on historical Bitcoin options data.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Running Backtests](#running-backtests)
4. [Understanding Results](#understanding-results)
5. [Advanced Usage](#advanced-usage)
6. [API Reference](#api-reference)

---

## Quick Start

### 1. Prepare Your Data

Ensure you have historical data files in your data directory:

```bash
sample_data/
├── btc_perpetual_20240101_20240131.parquet  # Required
└── btc_options_20240101_20240131.parquet    # Optional (for price comparison)
```

### 2. Create Configuration File

Create `config.yaml`:

```yaml
start_date: '2024-01-01'
end_date: '2024-01-31'
strike: 50000
maturity_days: 14
model_path: models/deep_hedger.pth
call: true
n_bootstrap_paths: 100
data_dir: sample_data
```

### 3. Run Backtest

```bash
python -m crypto.backtest.run --config config.yaml --seed 42
```

That's it! Results will be saved to `backtest_results/`.

---

## Configuration

### YAML Configuration File

**Recommended approach** for production use.

```yaml
# Required parameters
start_date: '2024-01-01'      # Backtest start date (YYYY-MM-DD)
end_date: '2024-01-31'        # Backtest end date (YYYY-MM-DD)
strike: 50000                 # Option strike price in USD
maturity_days: 14             # Option maturity in days
model_path: models/model.pth  # Path to trained model checkpoint

# Option type
call: true                    # true for call, false for put

# Execution parameters (optional - defaults shown)
n_bootstrap_paths: 100        # Number of bootstrap paths to generate
transaction_cost: 0.0005      # Transaction cost rate (0.05%)
dt_hours: 8.0                 # Rebalancing interval in hours
band_width: 0.001             # Minimum trade size in BTC (0.001=Binance, 0.01=Deribit, 0.0=disable)

# Directories (optional - defaults shown)
data_dir: sample_data         # Where to find historical data
output_dir: backtest_results  # Where to save results

# Execution parameters (optional - defaults shown)
seed: 42                      # Random seed for reproducibility
save_raw_data: false          # Save raw timeseries (positions, PnL, spots)
```

### Python API

```python
from crypto.backtest import BacktestConfig, Backtester

# Create config
config = BacktestConfig(
    start_date="2024-01-01",
    end_date="2024-01-31",
    strike=50000,
    maturity_days=14,
    model_path="models/deep_hedger.pth",
    call=True,
    n_bootstrap_paths=100,
)

# Run backtest
backtester = Backtester(config)
results = backtester.run(seed=42)

# Analyze results
summary = results.summary()
print(f"Deep Hedge Sharpe: {summary['deep']['sharpe']:.3f}")
print(f"BS Sharpe: {summary['bs']['sharpe']:.3f}")

# Generate report
results.generate_report("my_results")
```

### Saving/Loading Configs

```python
# Save config to YAML
config.save_yaml("configs/my_backtest.yaml")

# Load config from YAML
config = BacktestConfig.load_yaml("configs/my_backtest.yaml")
```

### Generating a Config Template

Don't want to write YAML from scratch? Generate a template:

```bash
# Print template to stdout
python -m crypto.backtest.run --print-default-config

# Save template to file
python -m crypto.backtest.run --print-default-config > my_config.yaml
```

The template includes:
- All required and optional parameters with descriptions
- Examples of path expansion (tilde, environment variables, relative paths)
- Inline comments explaining each option

Perfect for getting started quickly!

### Path Expansion

The framework supports flexible path specifications in your YAML configs:

**Tilde Expansion:**
```yaml
model_path: ~/models/deep_hedger.pth  # Expands to your home directory
```

**Environment Variables:**
```yaml
model_path: $HOME/models/model.pth          # Unix-style
model_path: ${MODEL_DIR}/deep_hedger.pth    # Bracketed syntax
```

**Relative Paths:**
```yaml
# Resolved from config file's directory, not current working directory
model_path: ../models/deep_hedger.pth
data_dir: ../data
```

**Absolute Paths:**
```yaml
model_path: /absolute/path/to/models/deep_hedger.pth
```

All path types can be mixed and matched as needed!

---

## Running Backtests

### Method 1: CLI with Config File (Recommended)

```bash
# Basic usage
python -m crypto.backtest.run --config config.yaml

# With seed in config file
python -m crypto.backtest.run --config config.yaml

# Override config parameters via CLI
python -m crypto.backtest.run --config config.yaml --n-paths 200 --seed 123

# Skip report generation (faster)
python -m crypto.backtest.run --config config.yaml --no-report

# Save raw timeseries data (can also set in config: save_raw_data: true)
python -m crypto.backtest.run --config config.yaml --save-raw-data
```

### Method 2: CLI Without Config File

```bash
python -m crypto.backtest.run \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --strike 50000 \
    --maturity-days 14 \
    --model-path models/deep_hedger.pth \
    --data-dir sample_data \
    --seed 42
```

### Method 3: Python Script

```python
from crypto.backtest import BacktestConfig, Backtester

config = BacktestConfig(
    start_date="2024-01-01",
    end_date="2024-01-31",
    strike=50000,
    maturity_days=14,
    model_path="models/deep_hedger.pth",
)

backtester = Backtester(config)
results = backtester.run(seed=42)
```

### Method 4: Jupyter Notebook

See `quick_test.ipynb` for interactive examples.

---

## Understanding Results

### Console Output

After running a backtest, you'll see:

```
==============================================================
BACKTEST RESULTS
==============================================================

Deep Hedge:
  Mean PnL: $1,234.56
  Std Dev: $567.89
  Sharpe Ratio: 2.174
  Max Drawdown: $-89.12
  CVaR (95%): $-123.45

Black-Scholes:
  Mean PnL: $987.65
  Std Dev: $678.90
  Sharpe Ratio: 1.454
  Max Drawdown: $-156.78
  CVaR (95%): $-234.56

Improvement:
  Mean PnL: $246.91 (25.0%)
  Sharpe Ratio: 0.720
==============================================================
```

### Generated Files

Results are saved to `output_dir/`:

```
backtest_results/
├── results.json                 # Summary statistics and config
├── backtest_report.md           # Markdown summary report
├── plots/
│   ├── pnl_comparison.png       # Cumulative PnL plot
│   ├── pnl_distribution.png     # PnL histogram
│   ├── positions.png            # Hedge positions over time
│   └── summary.png              # 2x2 grid of all plots
```

**Note:** By default, `results.json` contains only summary statistics. To save raw timeseries data (positions, PnL, spots) for custom analysis, either set `save_raw_data: true` in your config file or use the `--save-raw-data` CLI flag:

```bash
# Method 1: Set in config file
save_raw_data: true

# Method 2: Use CLI flag
python -m crypto.backtest.run --config config.yaml --save-raw-data
```

When `--save-raw-data` is enabled, `results.json` will include:
- `deep_pnl`: Deep hedge cumulative PnL over time (shape: `[n_paths, n_steps]`)
- `bs_pnl`: Black-Scholes cumulative PnL over time
- `deep_positions`: Deep hedge positions over time
- `bs_positions`: BS delta positions over time
- `spots`: Spot prices over time

**Warning:** Files with raw data can be large (e.g., 1000 paths × 100 steps × 5 arrays ≈ 5-10 MB)

### Metrics Explained

**Performance Metrics:**
- **Mean PnL**: Average profit/loss across all paths
- **Std Dev**: Standard deviation of PnL (risk measure)
- **Sharpe Ratio**: Risk-adjusted return (Mean / Std Dev)
- **Win Rate**: Percentage of paths with positive PnL

**Risk Metrics:**
- **CVaR (95%)**: Conditional Value at Risk - average loss in worst 5% of cases
- **VaR (95%)**: Value at Risk - 95th percentile loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Mean return / Max Drawdown

**Advanced Metrics:**
- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility

### Interpreting Plots

**1. PnL Comparison** (`pnl_comparison.png`):
- Shows cumulative PnL over time for both strategies
- Shaded regions show ±1 std dev across paths
- Higher line = better performance

**2. PnL Distribution** (`pnl_distribution.png`):
- Histogram of final PnL across all paths
- Right-shifted = better (more positive outcomes)
- Narrower = more consistent

**3. Hedge Positions** (`positions.png`):
- Shows hedge ratio (# of futures contracts) over time
- Deep hedge adapts dynamically, BS follows delta formula

**4. Comprehensive Analysis** (`comprehensive_analysis.png`):
- Combined view with metrics table
- Use for presentations/reports

---

## Determinism and Reproducibility

### Overview

The backtest framework provides comprehensive support for reproducible results across runs and machines. This is critical for:
- **Research**: Verify findings and share exact results with colleagues
- **Production**: Ensure consistent behavior in deployment
- **Debugging**: Isolate issues by eliminating randomness
- **Compliance**: Maintain audit trails for regulatory requirements

### Setting Random Seeds

The simplest way to ensure reproducibility is to set a random seed:

```bash
# CLI usage
python -m crypto.backtest.run --config config.yaml --seed 42
```

```python
# Python API usage
results = backtester.run(seed=42)
```

#### What Gets Seeded

When you provide a seed, the framework sets:
- **NumPy RNG**: For bootstrap sampling and data processing
- **PyTorch RNG**: For tensor operations and neural network inference
- **Python RNG**: For general random operations
- **CUDA RNG** (if GPU): For all CUDA operations on all devices

#### Implementation Details

The framework uses the following seeding strategy internally:

```python
import torch
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# CUDA seeds for GPU reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Enable deterministic mode for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### CPU vs GPU Reproducibility

**CPU (Fully Deterministic):**
- Results are **bitwise identical** across runs with the same seed
- No special configuration needed beyond setting seed
- Recommended for research and development

**GPU (Mostly Deterministic):**
- Results are **numerically equivalent** but may have minor floating-point differences
- Requires additional cuDNN configuration (handled automatically)
- Some operations may still be non-deterministic due to hardware/driver variations
- Trade-off: Deterministic mode disables some optimizations (10-30% slower)

**Performance Impact:**
```
Setting                           Speed    Determinism
────────────────────────────────  ───────  ──────────────
CPU (with seed)                   Baseline 100% identical
GPU (benchmark mode)              Fastest  Non-deterministic
GPU (deterministic mode)          -20%     ~99.99% identical
```

### Provenance Tracking

The framework automatically captures provenance information for full reproducibility:

```python
from crypto.backtest import BacktestConfig

# Load config with automatic path resolution
config = BacktestConfig.load_yaml("config.yaml")

# Get provenance information
provenance = config.get_provenance_info()

# Provenance includes:
# - Full configuration (all parameters)
# - Resolved absolute paths (data_dir, model_path, output_dir)
# - Git commit hash (if in git repo)
# - Git branch name
# - Git dirty status (uncommitted changes)
# - Python version
# - Platform/OS
# - Timestamp
```

**Provenance Hash** (coming soon):
Each backtest run generates a hash that uniquely identifies:
- Configuration parameters
- Data source (file paths + modification times)
- Model checkpoint (file path + hash)
- Software version (git commit)

This hash is included in report headers for easy reference.

### Configuration Management

**Best Practice**: Store configs in version control

```bash
git_repo/
├── configs/
│   ├── prod_baseline.yaml      # Production baseline
│   ├── experiment_v1.yaml      # Experiment variant
│   └── validation_202501.yaml  # Validation run
├── results/
│   └── [auto-generated]/       # Results reference config files
└── models/
    └── deep_hedger_v3.pth
```

**YAML configs are anchored to their directory:**
```yaml
# If this config is in configs/backtest.yaml
# Then relative paths resolve from configs/ directory

model_path: ../models/deep_hedger.pth       # Resolves to models/deep_hedger.pth
data_dir: ../data                           # Resolves to data/
output_dir: ../results/experiment_001       # Resolves to results/experiment_001/
```

**Environment-based configs:**
```yaml
# Development
model_path: $HOME/dev/models/deep_hedger.pth
data_dir: $PROJECT_ROOT/data

# Production
model_path: /mnt/models/production/deep_hedger_v3.pth
data_dir: /mnt/data/bitcoin
```

### Reproducibility Checklist

To guarantee identical results across runs:

- [ ] **Set random seed** via `--seed` or `run(seed=N)`
- [ ] **Pin software versions** (Python, PyTorch, NumPy)
- [ ] **Track git commit** (automatic if in git repo)
- [ ] **Save YAML config** for each experiment
- [ ] **Document device** (CPU vs GPU model)
- [ ] **Record PyTorch version** and CUDA version (if GPU)
- [ ] **Verify data integrity** (same data files)
- [ ] **Check model checkpoint** (same .pth file)

### Dealing with Non-Determinism

**If results differ despite setting seed:**

1. **Check data files**: Ensure exact same data files (checksum)
2. **Check model checkpoint**: Verify .pth file is identical
3. **Check PyTorch version**: Minor version differences can cause variance
4. **Check device**: CPU vs GPU can produce different results
5. **Check cuDNN version**: GPU driver updates can affect behavior
6. **Check floating-point mode**: Some operations use FP16/BF16

**Debugging workflow:**
```python
# 1. Run with seed, save config
config.save_yaml("debug_config.yaml")
results = backtester.run(seed=42)

# Save with raw data for detailed analysis
results.to_json("results_full.json", include_raw=True)

# Or just summary stats (smaller file)
results.to_json("results_summary.json", include_raw=False)

# 2. Re-run from saved config
config2 = BacktestConfig.load_yaml("debug_config.yaml")
results2 = Backtester(config2).run(seed=42)

# 3. Compare results (should be identical on same machine)
assert torch.allclose(results.deep_pnl, results2.deep_pnl)
```

### Sharing Results

**For collaborators:**
```bash
# Share these files
configs/experiment.yaml       # Exact configuration
results/report.md             # Results summary (includes provenance)
models/deep_hedger_v3.pth     # Model checkpoint

# They can reproduce with:
python -m crypto.backtest.run --config configs/experiment.yaml --seed 42
```

**For publications/papers:**
Include in supplementary materials:
- YAML config file
- Git commit hash (or full code snapshot)
- Python/PyTorch/NumPy versions
- Random seed used
- Data file checksums (MD5/SHA256)
- Model checkpoint hash

### Advanced: Deterministic Testing

The framework includes deterministic regression tests:

```python
# See crypto/tests/test_backtest_reproducibility.py
def test_reproducibility():
    """Verify backtest results are identical across runs."""
    config = BacktestConfig(...)

    # Run 1
    results1 = Backtester(config).run(seed=42)

    # Run 2
    results2 = Backtester(config).run(seed=42)

    # Should be identical (bitwise on CPU)
    assert torch.equal(results1.deep_pnl, results2.deep_pnl)
    assert torch.equal(results1.bs_pnl, results2.bs_pnl)
```

### Platform-Specific Notes

**macOS (Apple Silicon M1/M2/M3):**
- Use `device='mps'` for Metal Performance Shaders acceleration
- MPS may have non-deterministic operations (use CPU for exact reproducibility)

**Linux (NVIDIA GPU):**
- Recommended for production (best deterministic support)
- Use `CUDA_LAUNCH_BLOCKING=1` for debugging non-determinism

**Windows:**
- Fully supported on CPU
- GPU support requires CUDA-enabled PyTorch installation

---

## Advanced Usage

### Comparing Multiple Strategies

```python
from crypto.backtest import BacktestConfig, Backtester

# Test different transaction costs
configs = [
    BacktestConfig(..., transaction_cost=0.0001),
    BacktestConfig(..., transaction_cost=0.0005),
    BacktestConfig(..., transaction_cost=0.001),
]

results = []
for config in configs:
    backtester = Backtester(config)
    results.append(backtester.run(seed=42))

# Compare Sharpe ratios
for i, r in enumerate(results):
    summary = r.summary()
    print(f"Cost {configs[i].transaction_cost:.4f}: "
          f"Sharpe = {summary['deep']['sharpe']:.3f}")
```

### Price Comparison with Real Options

```python
from crypto.backtest.option_comparison import OptionMatcher, PriceComparator

# Run backtest
results = backtester.run(seed=42)

# Load real option data
matcher = OptionMatcher(backtester.data_loader)

# Compare prices
comparator = PriceComparator(results, matcher)
comparison = comparator.compare_with_market(
    strike=50000,
    maturity_days=14,
    call=True,
)

print(f"Model Price: ${comparison['model_price']:.2f}")
print(f"Market Price: ${comparison['market_price']:.2f}")
print(f"Difference: {comparison['difference_pct']:.1f}%")
print(f"Model in Spread: {comparison['model_in_spread']}")
```

### IV Analysis

```python
# Compare implied volatilities
iv_comparison = comparator.compare_implied_volatility(
    strike=50000,
    maturity_days=14,
    call=True,
    spot_price=50000,
)

print(f"Model IV: {iv_comparison['model_iv']:.2%}")
print(f"Market IV: {iv_comparison['market_iv']:.2%}")
print(f"IV Difference: {iv_comparison['iv_difference_pct']:.1f}%")

# Generate volatility smile
strikes = [45000, 47500, 50000, 52500, 55000]
smile = comparator.get_volatility_smile(
    strikes=strikes,
    maturity_days=14,
    call=True,
    spot_price=50000,
)

print(smile[['strike', 'moneyness', 'model_iv', 'market_iv']])
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt

# Access raw data
deep_pnl = results.deep_pnl.cpu().numpy()  # Shape: (n_paths, n_steps)
bs_pnl = results.bs_pnl.cpu().numpy()

# Create custom plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(deep_pnl.mean(axis=0), label='Deep Hedge', linewidth=2)
ax.plot(bs_pnl.mean(axis=0), label='BS Baseline', linewidth=2)
ax.set_xlabel('Time Step')
ax.set_ylabel('Mean Cumulative PnL ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('custom_pnl_plot.png', dpi=300)
```

### Loading and Analyzing Saved Raw Data

```python
import json
import torch
import numpy as np

# Load saved results with raw data
with open('results.json') as f:
    data = json.load(f)

# Convert back to tensors
deep_pnl = torch.tensor(data['deep_pnl'])        # Shape: [n_paths, n_steps]
bs_pnl = torch.tensor(data['bs_pnl'])
deep_positions = torch.tensor(data['deep_positions'])
bs_positions = torch.tensor(data['bs_positions'])
spots = torch.tensor(data['spots'])

# Example: Calculate custom metrics
final_pnl = deep_pnl[:, -1]
print(f"Mean final PnL: ${final_pnl.mean():.2f}")
print(f"Median final PnL: ${final_pnl.median():.2f}")

# Example: Analyze worst-case path
worst_path_idx = final_pnl.argmin()
print(f"Worst path final PnL: ${final_pnl[worst_path_idx]:.2f}")

# Plot worst path evolution
import matplotlib.pyplot as plt
plt.plot(deep_pnl[worst_path_idx].numpy(), label='Deep Hedge (worst)')
plt.plot(bs_pnl[worst_path_idx].numpy(), label='BS (worst)')
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Time Step')
plt.ylabel('Cumulative PnL ($)')
plt.legend()
plt.title('Worst Path Analysis')
plt.savefig('worst_path.png')
```

---

## API Reference

### Core Classes

#### `BacktestConfig`

Configuration dataclass for backtests.

**Methods:**
- `validate()`: Validate all parameters
- `to_dict()`: Convert to dictionary
- `from_dict(config_dict)`: Create from dictionary
- `save_yaml(path)`: Save to YAML file
- `load_yaml(path)`: Load from YAML file

#### `Backtester`

Main backtesting engine.

**Methods:**
- `load_model()`: Load pre-trained deep hedging model
- `load_data()`: Load historical data from parquet files
- `create_bootstrap_option()`: Generate bootstrap paths from historical data
- `run_deep_hedge()`: Execute deep hedging strategy
- `run_bs_baseline()`: Execute Black-Scholes delta hedging
- `run(seed=None)`: Run complete backtest pipeline

**Returns:** `BacktestResults` object

#### `BacktestResults`

Container for backtest results and analysis.

**Properties:**
- `n_paths`: Number of paths
- `n_steps`: Number of time steps
- `deep_pnl`: Deep hedge PnL tensor (n_paths × n_steps)
- `bs_pnl`: BS baseline PnL tensor (n_paths × n_steps)
- `deep_positions`: Deep hedge positions
- `bs_positions`: BS positions
- `spots`: Spot price paths
- `config`: Configuration used

**Methods:**
- `summary()`: Calculate comprehensive metrics
- `to_dict(include_raw=True)`: Export to dictionary (optionally with raw timeseries)
- `to_json(filepath, include_raw=False)`: Save to JSON file (set `include_raw=True` to save position/PnL data)
- `generate_report(output_dir)`: Create markdown report with plots
- `plot_pnl_comparison()`: Plot cumulative PnL
- `plot_pnl_distribution()`: Plot PnL histogram
- `plot_positions()`: Plot hedge positions
- `plot_all()`: Create comprehensive 2x2 plot grid

#### `OptionMatcher`

Match backtest parameters to real market options.

**Methods:**
- `find_matching_options(strike, maturity_days, call)`: Find all matching options
- `get_closest_match(strike, maturity_days, call)`: Get single best match
- `get_time_series(strike, maturity_days, call, start_date, end_date)`: Track option over time
- `summary()`: Get data summary statistics

#### `PriceComparator`

Compare model-implied prices with market prices.

**Methods:**
- `calculate_model_implied_price()`: Get model's fair value estimate
- `calculate_model_price_confidence(confidence_level=0.95)`: Bootstrap confidence intervals
- `get_market_price(strike, maturity_days, call)`: Extract market price
- `compare_with_market(strike, maturity_days, call)`: Full price comparison with spread diagnostics
- `calculate_model_implied_iv(strike, maturity_days, call)`: Model-implied volatility
- `get_market_iv(strike, maturity_days, call)`: Market implied volatility
- `compare_implied_volatility(strike, maturity_days, call)`: Compare model vs market IV
- `get_volatility_smile(strikes, maturity_days, call)`: Generate volatility smile across strikes

---

## Troubleshooting

### Common Issues

**1. "Config file not found"**
- Check path is correct: `ls config.yaml`
- Use absolute path if needed: `--config /full/path/to/config.yaml`

**2. "Model file not found"**
- Verify model path in config
- Check file exists: `ls models/deep_hedger.pth`

**3. "No data files found"**
- Check data directory: `ls sample_data/`
- Ensure perpetual data exists: `ls sample_data/*perpetual*.parquet`

**4. "PyYAML required"**
- Install with: `pip install pyyaml`

**5. "Out of memory"**
- Reduce `n_bootstrap_paths` (try 50 or 20)
- Use smaller date range

**6. "Results look wrong"**
- Always use `--seed` for reproducibility
- Check config dates match data availability
- Verify model was trained on similar parameters

### Getting Help

1. Check examples in `quick_test.ipynb`
2. Review test files in `crypto/tests/`
3. Read docstrings: `help(Backtester.run)`
4. Check implementation plan: `BACKTEST_PLAN.md`

---

## Best Practices

1. **Always use `--seed`** for reproducibility
2. **Start small**: Test with 10-20 paths before scaling to 100+
3. **Save configs**: Keep YAML files for each experiment
4. **Version control**: Track configs, not just code
5. **Check data quality**: Verify date ranges before backtesting
6. **Compare apples to apples**: Use same config for strategy comparisons
7. **Monitor resources**: Large path counts can use significant memory
8. **Validate results**: Cross-check with option pricing theory

---

## Examples

### Example 1: Simple Call Option Backtest

```bash
python -m crypto.backtest.run \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --strike 50000 \
    --maturity-days 14 \
    --call \
    --model-path models/deep_hedger.pth \
    --data-dir sample_data \
    --seed 42
```

### Example 2: Put Option with Custom Parameters

```yaml
# put_config.yaml
start_date: '2024-01-01'
end_date: '2024-02-29'
strike: 45000
maturity_days: 30
model_path: models/deep_hedger.pth
call: false
n_bootstrap_paths: 200
transaction_cost: 0.001
dt_hours: 4.0
```

```bash
python -m crypto.backtest.run --config put_config.yaml --seed 123
```

### Example 3: Batch Processing

```python
import pandas as pd
from crypto.backtest import BacktestConfig, Backtester

# Parameter grid
strikes = [45000, 50000, 55000]
maturities = [7, 14, 30]

results_df = []
for strike in strikes:
    for maturity in maturities:
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=strike,
            maturity_days=maturity,
            model_path="models/deep_hedger.pth",
            n_bootstrap_paths=50,
        )

        backtester = Backtester(config)
        results = backtester.run(seed=42)
        summary = results.summary()

        results_df.append({
            'strike': strike,
            'maturity': maturity,
            'deep_sharpe': summary['deep']['sharpe'],
            'bs_sharpe': summary['bs']['sharpe'],
            'improvement': summary['deep']['sharpe'] - summary['bs']['sharpe'],
        })

# Analyze results
df = pd.DataFrame(results_df)
print(df.to_string(index=False))
```

---

## See Also

- `BACKTEST_PLAN.md` - Implementation roadmap
- `PHASE3_SUMMARY.md` - Option price comparison details
- `quick_test.ipynb` - Interactive examples
- `crypto/examples/` - Training examples

---

*Last updated: 2025-10-18*
*Framework version: Phase 4 Complete*
