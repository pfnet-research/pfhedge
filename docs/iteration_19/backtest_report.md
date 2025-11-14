# Backtest Report
**Generated:** 2025-11-11 23:23:04
**Paths:** 1,000 | **Steps:** 100

## Configuration
- **start_date**: 2025-01-01
- **end_date**: 2025-09-25
- **strike**: 110000
- **maturity_days**: 33
- **model_path**: /workspace/pfhedge/models/iteration_19_87b30298/model.pth
- **call**: True
- **n_bootstrap_paths**: 1000
- **transaction_cost**: 0.0004
- **dt_hours**: 8.0
- **volatility_window**: 20
- **underlying_type**: perpetual
- **band_width**: 0.001
- **data_dir**: /workspace/pfhedge/crypto/data/historical
- **data_file**: btc_spot_8H_2025-01-01_2025-10-28.parquet
- **output_dir**: /workspace/pfhedge/backtest_results
- **bootstrap_mode**: absolute_strike
- **initial_spot**: None
- **target_moneyness**: None
- **spot_tolerance**: 0.1
- **enable_diagnostics**: False
- **seed**: 42
- **save_raw_data**: True

## Performance Summary

### Deep Hedge
- **Mean PnL**: $-3,522.55
- **Std PnL**: $3,664.91
- **Sharpe Ratio**: -0.961
- **Sortino Ratio**: -0.702
- **CVaR (95%)**: $-11,248.69
- **Max Drawdown**: $6,839.58
- **Win Rate**: 17.7%

### Black-Scholes Baseline
- **Mean PnL**: $-4,259.29
- **Std PnL**: $3,999.64
- **Sharpe Ratio**: -1.065
- **Sortino Ratio**: -0.729
- **CVaR (95%)**: $-12,669.11
- **Max Drawdown**: $7,289.83
- **Win Rate**: 8.6%

### Comparison (Deep - BS)
- **Mean PnL Diff**: $+736.75
- **Sharpe Diff**: +0.104
- **CVaR Diff**: $+1,420.42

## Visualizations

### PnL Comparison
![PnL Comparison](plots/pnl_comparison.png)

### PnL Distribution
![PnL Distribution](plots/pnl_distribution.png)

### Hedge Positions
![Positions](plots/positions.png)

### Summary
![Summary](plots/summary.png)
