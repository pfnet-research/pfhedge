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