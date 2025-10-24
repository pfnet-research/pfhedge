## Coding Standards

* **Minimalist principle**: Only keep code that is actually necessary. Don't write code that won't be used right now.
* **Follow PFHedge patterns**: When implementing deep hedging, follow established patterns from `examples/snowball_hedge.py`
  - Use standard PFHedge features: `log_moneyness`, `expiry_time`, `volatility`, `prev_hedge`
  - Use correct methods: `compute_hedge()`, `compute_cum_pl()`, `cum_pl()`
  - Don't extract custom features unless specifically needed
* **DON'T Apologize in reply**

## Workflow

* use @crypto/plan.md to keep track of crypto hedging engine progress
* for each task, plan it first and figure out how to test. Then implement it until test pass
* use @quick_test.ipynb as a running example. Whenever you implement a new feature, add it to the quick test
* @quick_test.ipynb now includes complete deep hedging training demo (sections 10-12) - train model, generate strategies, evaluate performance

## Lessons

**From the `get_recent_trades()` bug (2025-10):**

* **Always validate temporal data**: For time-series/trading data, assert that timestamps match expectations. Don't just check if data exists - verify it's the RIGHT data from the RIGHT time.
* **Test data correctness, not just existence**: Tests must verify VALUES are correct, not just that fields are present. Example: `assert trade['timestamp'] >= start_time and trade['timestamp'] <= end_time`
* **Beware of "convenience" methods**: Methods that hide important parameters (like date/time) are dangerous in financial applications. Prefer explicit: `get_historical_trades(start, end)` over `get_recent_trades()`
* **Explicit over implicit**: Make temporal dependencies visible in method signatures. If a method assumes "yesterday", that assumption should be in the name or require an explicit date parameter.
* **Add critical assertions in production code**: Data fetching functions should validate their results match requested parameters before returning
* **Integration tests need temporal assertions**: End-to-end tests passed because we checked "got trades" but not "got trades FROM THE CORRECT DATE". Always verify temporal correctness.

**From Bootstrap Moneyness Consistency (2025-01):**

* **Problem**: Original bootstrap used fixed strike with varying historical spot prices, causing each path to test fundamentally different options (e.g., deep OTM in Jan vs ATM in Oct). This made backtest statistics meaningless for forward-looking trading decisions.
* **Root cause**: Neural network received different `log_moneyness` inputs across paths, far outside training distribution. Averaging P&L from different option types (OTM/ATM/ITM) gave meaningless results.
* **Solution**: Added `bootstrap_mode` with spot rescaling:
  - `normalize_spot`: Rescales historical prices to preserve target moneyness (recommended for trading decisions)
  - `absolute_strike`: Legacy mode for backward compatibility (raw prices, varying moneyness)
* **Key insights**:
  1. **Multiplicative rescaling preserves market dynamics**: Rescaling spot prices by a constant factor preserves returns and volatility (tested)
  2. **Rescale ALL price columns; preserve rates/quantities**: Use whitelist for price-like columns (`last_price`, `bid_price`, etc.), but don't touch `funding_rate`, volumes
  3. **Store scale_factors for auditability**: Track rescaling for each path to verify correctness
  4. **Enforce no look-ahead**: Sample windows strictly from data < trade_date using `max_date` parameter
  5. **Single source of truth for n_steps**: Use consistent formula `int(time_horizon / dt) + 1` to avoid off-by-one errors
* **Implementation pattern**: Fix in bootstrap logic (rescale spots), not in option class (which stays simple with scalar strike)
* **Testing priorities**:
  - Returns invariance under rescaling
  - Initial moneyness equality across paths
  - Backward compatibility with `absolute_strike` mode
  - Funding cost proportional scaling
  - Edge cases (single window, insufficient data)
* **User communication**: Clear defaults (`absolute_strike` for backward compat), helpful error messages for `normalize_spot` requiring moneyness specification