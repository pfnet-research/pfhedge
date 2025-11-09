# Hyperparameter Tuning Analysis Scripts

A comprehensive suite of scripts to analyze hyperparameter tuning results and identify the best models.

## Quick Start

Run the complete analysis pipeline:

```bash
python crypto/scripts/run_hparam_analysis.py
```

This will:
1. Aggregate all model results from `hparam_tuning/`
2. Compare and rank models by various metrics
3. Generate comprehensive visualizations
4. Create detailed HTML/Markdown reports

View the results:
```bash
open hparam_tuning/analysis/report.html
```

## Individual Scripts

### 1. Aggregate Results
```bash
python crypto/scripts/analyze_hparam_results.py \
    --hparam-dir hparam_tuning \
    --output-dir hparam_tuning/analysis
```

**Output:**
- `all_results.csv` - Human-readable CSV with all metrics
- `all_results.pkl` - Pickle file for Python analysis

**What it does:**
- Parses hyperparameters from directory names
- Loads training results (loss, test metrics)
- Loads backtest results (PnL, Sharpe, CVaR, etc.)
- Combines into single DataFrame with 108 models × 47 metrics

### 2. Compare & Rank Models
```bash
python crypto/scripts/compare_hparam_models.py \
    --results-file hparam_tuning/analysis/all_results.pkl \
    --output-dir hparam_tuning/analysis/rankings \
    --top-n 10
```

**Output:**
- `top_10_by_*.csv` - Top performers for each metric
- `best_vs_baseline.csv` - Comparison with Black-Scholes
- `pareto_sharpe_vs_cvar.csv` - Pareto-optimal models
- `performance_by_architecture.csv` - Average by model type
- `impact_of_*.csv` - Hyperparameter sensitivity analysis

**What it does:**
- Ranks models by multiple criteria
- Finds Pareto frontier
- Compares deep hedge vs Black-Scholes baseline
- Analyzes hyperparameter impacts
- Recommends best model

### 3. Generate Visualizations
```bash
python crypto/scripts/visualize_hparam_results.py \
    --results-file hparam_tuning/analysis/all_results.pkl \
    --output-dir hparam_tuning/analysis/plots \
    --top-n 10
```

**Output:** (13 plots)
- `metric_distributions.png` - Distribution of key metrics
- `performance_by_architecture.png` - MLP vs LSTM vs GRU
- `dh_sharpe_ratio_by_*.png` - Impact of hyperparameters
- `heatmap_*.png` - 2D parameter analysis
- `correlation_matrix.png` - Metric correlations
- `deep_vs_baseline.png` - Deep hedge vs BS comparison
- `risk_return_tradeoff.png` - Risk-return scatter
- `top_10_models_comparison.png` - Top performers
- `risk_measure_comparison.png` - Entropic vs ES

**What it does:**
- Creates comprehensive visualizations
- Compares architectures and hyperparameters
- Analyzes risk-return tradeoffs
- Visualizes top performers

### 4. Generate Report
```bash
python crypto/scripts/generate_hparam_report.py \
    --results-file hparam_tuning/analysis/all_results.pkl \
    --rankings-dir hparam_tuning/analysis/rankings \
    --plots-dir hparam_tuning/analysis/plots \
    --output-dir hparam_tuning/analysis
```

**Output:**
- `report.html` - Interactive HTML report
- `report.md` - Markdown report

**What it includes:**
- Executive summary with best model
- Performance overview and statistics
- Architecture comparison
- Hyperparameter analysis
- Deep hedge vs Black-Scholes comparison
- Top performers
- Training insights
- Key findings and recommendations

## Output Structure

```
hparam_tuning/analysis/
├── all_results.csv              # All results (human-readable)
├── all_results.pkl              # All results (Python)
├── report.html                  # Main report (open this!)
├── report.md                    # Markdown version
├── rankings/                    # 14 ranking CSV files
│   ├── top_10_by_dh_sharpe_ratio.csv
│   ├── top_10_by_dh_mean.csv
│   ├── best_vs_baseline.csv
│   ├── pareto_sharpe_vs_cvar.csv
│   └── ...
└── plots/                       # 13 visualization PNGs
    ├── metric_distributions.png
    ├── performance_by_architecture.png
    ├── deep_vs_baseline.png
    └── ...
```

## Key Metrics

### Primary Metrics (for model selection)
- **Sharpe Ratio** - Risk-adjusted returns (higher is better)
- **Mean PnL** - Average profit/loss (higher is better)
- **CVaR 95%** - Tail risk (higher/less negative is better)
- **Sharpe Improvement** - vs Black-Scholes baseline

### Secondary Metrics
- **Sortino Ratio** - Downside risk-adjusted returns
- **Win Rate** - Percentage of profitable paths
- **Max Drawdown** - Worst cumulative loss
- **Calmar Ratio** - Return/max drawdown

## Advanced Usage

### Custom analysis directory
```bash
python crypto/scripts/run_hparam_analysis.py \
    --hparam-dir results/my_tuning \
    --output-dir results/my_analysis \
    --top-n 20
```

### Skip already-completed steps
```bash
python crypto/scripts/run_hparam_analysis.py \
    --skip-aggregation \
    --skip-comparison
```

### Run only specific steps
```bash
# Only generate visualizations
python crypto/scripts/visualize_hparam_results.py

# Only generate report
python crypto/scripts/generate_hparam_report.py
```

## Dependencies

All scripts use standard libraries:
- pandas
- numpy
- matplotlib
- seaborn

## Troubleshooting

**Error: "Results file not found"**
- Run aggregation step first: `python crypto/scripts/analyze_hparam_results.py`

**Empty results**
- Check that `hparam_tuning/` contains model directories
- Verify directory names match pattern: `{hash}_{model_type}_l{layers}_u{units}_{risk}{param}_lr{lr}`
- Ensure each directory has `train_*/` and `backtest_*/` subdirectories

**Missing plots in report**
- Run visualization step before report generation
- Check that `hparam_tuning/analysis/plots/` exists

## Example Output

**Best Model (from 108 models):**
- ID: `4a4f3754f7_lstm_l3_u32_entropic1.0_lr0.0001`
- Architecture: LSTM, 3 layers, 32 units
- Risk: Entropic risk (α=1.0)
- Sharpe: -0.6525 (best among all models)
- Improvement: +3.02 vs Black-Scholes baseline

**Key Findings:**
- LSTM models performed best on average
- 2-layer architectures slightly outperformed 3-layer
- 32 units showed best results (less overfitting)
- Entropic risk marginally better than Expected Shortfall
- 100% of models beat Black-Scholes on Sharpe ratio
- Only 2% beat Black-Scholes on CVaR (tail risk remains challenging)
