## Project Map

### Core Framework
- `crypto/instruments/` - Bitcoin option and underlier instruments (spot/perpetual Brownian motion)
- `crypto/strategies/` - Hedging strategies (deep hedge models, BS baseline, no-trade band utils)
- `crypto/training/` - Training framework (Trainer, TrainingConfig, results)
- `crypto/backtest/` - Backtesting framework (Backtester, StrategyExecutor, configs)

### Scripts & Tools
- `crypto/scripts/` - All executable scripts (see crypto/scripts/README.md for complete reference)
  - **Core workflow:** explore_options.py, train_for_option.py, calculate_seller_pnl.py, fetch_deribit_data.py
  - **Hyperparameter tuning:** tune_for_option.py, run_hparam_analysis.py, analyze/compare/visualize/generate_hparam_*.py
  - **Diagnostics:** verify_tardis.py, diagnose_gpu.py, profile_training.py, debug_pnl.py
  - **See crypto/scripts/README.md for detailed description of what each script does**
- `crypto/data/` - Historical data download and processing (TARDIS API integration)

### Tests & Docs
- `crypto/tests/` - Unit tests (feature leakage, GBM sanity, baselines, no-trade band)
- `crypto/backtest/USAGE.md` - Backtesting guide
- `crypto/training/USAGE.md` - Training guide
- `crypto/docs/INSTRUCTIONS.md` - Quick 4-step workflow guide (data → explore → train → backtest)
- `crypto/docs/STEP_BY_STEP_WORKFLOW.md` - Detailed interactive workflow with decision points
- `docs/Real-World-Deep-Hedging-Tricks.md` - Implementation tricks and best practices

### Key Files
- `backtest.yaml` - Backtest configuration template
- `crypto/strategies/deep_hedge_utils.py` - Core utilities (PnL calculation, no-trade band)
- `crypto/training/trainer.py` - Main training loop and model evaluation

## Finding Script Information

**IMPORTANT:** To understand what any script in `crypto/scripts/` does, ALWAYS check `crypto/scripts/README.md` first.

The README contains comprehensive documentation for all 18 scripts:
- What each script does
- Usage examples with command-line flags
- Expected inputs and outputs
- When to use each script

Never guess what a script does - read the README.md reference.

## Workflow
* TARDIS_API_KEY=TD.rSzoJCymVt13xucv.f0uh1Crgt0pdOcz.jyqhwMa1PoSjzEo.Xwjrec5DEivk4ON.Sgyf7c169jr4RCu.sXMR
* for each task, plan it first and figure out how to test. Then implement it until test pass
* 