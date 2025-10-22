# Trading Scripts

Interactive and automated tools for option trading with deep hedging.

## Documentation

📖 **Complete guide:** See [`crypto/docs/STEP_BY_STEP_WORKFLOW.md`](../docs/STEP_BY_STEP_WORKFLOW.md)

This includes:
- Quick reference table of all scripts
- Step-by-step interactive workflow
- Complete examples
- Troubleshooting
- Tips for real trading

## Quick Overview

### Interactive Workflow (4 steps with human decisions)
1. `explore_options.py` - Find and review available options
2. `train_for_option.py` - Train model for selected option
3. `crypto.backtest.run` - Backtest the strategy
4. `calculate_seller_pnl.py` - Calculate final expected returns

### Automated Workflow (for batch jobs)
- `realistic_backtest.py` - End-to-end automation without human review

### Supporting Scripts
- `fetch_deribit_data.py` - Download historical data from Deribit
- `select_option.py` - Programmatic option selection (used internally)

---

**For detailed usage, examples, and complete workflow guide:**
👉 [`crypto/docs/STEP_BY_STEP_WORKFLOW.md`](../docs/STEP_BY_STEP_WORKFLOW.md)
