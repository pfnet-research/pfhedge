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