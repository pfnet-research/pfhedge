# Revised Deep Hedging Improvement Plan: Risk-Aware, Gated, Production-Ready

## Executive Summary
This plan incorporates critical feedback from senior AI researchers to create a robust, de-risked approach to improving deep hedging performance. The core principle: **verify before scale, gate before advance, measure comprehensively before claiming success.**

---

## Phase 0: Pre-Flight Infrastructure & Readiness (Days 1-2)

### 0.1 Code Readiness Check
```python
# Verify crypto.scripts.train_for_option supports:
required_features = {
    '--model-type': ['mlp', 'lstm', 'gru'],
    '--objective': ['entropic', 'cvar', 'variance'],
    '--risk-param': 'float',
    '--use-iv-surface': 'bool',
    '--seed': 'int',
    '--log-dir': 'path',
}

# If missing, add to Phase 0 implementation
```

### 0.2 Reproducibility Infrastructure
```python
logging_config = {
    'seed': 42,
    'git_commit': get_git_hash(),
    'config_hash': hash_config(config),
    'output_dir': 'backtest_results/{timestamp}_{git_hash[:8]}/',
    'save_format': 'results.json',
    'track_gradients': True,
    'gradient_clip_norm': 1.0,
}
```

### 0.3 Feature Leakage Audit & Unit Test
```python
def test_no_future_leakage():
    """Critical: Ensure features are strictly past-only"""
    # Construct features at time t using data[:t]
    features_normal = construct_features(data, t)

    # Shift labels by +1 step
    features_shifted = construct_features(data, t+1)

    # Train on shifted - should collapse
    model_shifted = train(features_shifted, labels[:-1])
    perf_shifted = evaluate(model_shifted)

    assert perf_shifted < baseline * 0.5, "Leakage detected!"
```

### 0.4 Baseline Suite Verification
```python
# Verify existing baselines work (already implemented in crypto/)
baselines = {
    'black_scholes_delta': run_bs_baseline(),  # Already exists in strategy_executor.py
    'no_hedge': lambda: 0,  # Trivial: never hedge
    'unhedged': lambda: 0,  # For variance comparison (same as no_hedge)
}

# Optional: Whalley-Wilmott from pfhedge (if time permits)
# baselines['whalley_wilmott'] = WhalleyWilmott()
```

**Gate G0: Cannot proceed without:**
- ✓ Code supports all required features
- ✓ Reproducibility infrastructure operational
- ✓ Leakage test passes
- ✓ All baselines implemented and tested
- ✓ Logging outputs to versioned directory

---

## Phase 1: GBM Simulator Sanity Check (Day 3)

**Current Setup:** Your implementation uses `BitcoinSpotBrownian` and `BitcoinPerpetualBrownian` with `mu=0.0` (risk-neutral by construction). No drift contamination is possible with pure GBM.

### 1.1 Quick GBM Verification (30 minutes)
```python
def test_gbm_sanity():
    """Verify GBM implementation works correctly"""
    from crypto.instruments import BitcoinSpotBrownian

    simulator = BitcoinSpotBrownian(mu=0.0, sigma=0.8, dt=8/24/365)
    paths = simulator.simulate(n_paths=10000, n_steps=42)

    S0 = paths[:, 0].mean()
    ST = paths[:, -1].mean()
    ratio = ST / S0

    # Should be ~1.0 within statistical noise (2 std errors)
    std_error = paths[:, -1].std() / np.sqrt(10000) / S0

    assert abs(ratio - 1.0) < 2 * std_error, (
        f"GBM martingale check failed: E[ST]/S0 = {ratio:.4f}, "
        f"expected ~1.0 ± {2*std_error:.4f}"
    )

    print(f"✓ GBM sanity check passed: E[ST]/S0 = {ratio:.4f}")
    # Runtime: ~10 seconds
```

### 1.2 Document Current Simulator (30 minutes)
```python
# Document in crypto/docs/simulator_config.md
"""
Current Simulator: Geometric Brownian Motion
- Implementation: BitcoinSpotBrownian (crypto/instruments/)
- Risk-neutral: mu=0.0 (hardcoded)
- Volatility: sigma=0.8 (configurable)
- Time step: dt=8/24/365 (8-hour bars)
- Transaction cost: 0.05-0.06% (configurable)

This is a simplified simulator for rapid iteration.
"""
```

**Gate G1: Cannot proceed without:**
- ✓ GBM sanity check passes (10 seconds runtime)
- ✓ mu=0.0 documented and verified in code
- ✓ No drift contamination possible (guaranteed by construction)

**Note:** Detailed martingale testing, drift detection, and risk-neutralization will become critical when you implement GAN-based simulators (see Appendix: Future Work)

---

## Phase 2: Small-Scale Architecture Search (Days 8-12)

### 2.1 Architecture Comparison with Engineering Guardrails
```python
architectures = {
    'mlp_simple': {
        'type': 'MLP',
        'layers': [32, 32],
        'dropout': 0.2
    },
    'gru_light': {  # Start with GRU - often more stable
        'type': 'GRU',
        'hidden_size': 64,
        'num_layers': 1,
        'dropout': 0.2,
        'recurrent_dropout': 0.1
    },
    'lstm_standard': {
        'type': 'LSTM',
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'layer_norm': True
    },
}

training_config = {
    'paths': 50_000,  # Small for speed
    'epochs': 100,
    'early_stopping_patience': 15,
    'gradient_clip_norm': 1.0,  # Critical for RNNs
    'mixed_precision': True,     # Speed up
    'lr_scheduler': 'reduce_on_plateau',
    'max_wall_time_minutes': 30, # Auto-stop
}
```

### 2.2 Objective Function Strategy (Stable Start)
```python
# Start with stable entropic, not noisy CVaR
objectives = {
    'entropic_tuned': EntropicRisk(lambda_=0.1),  # Tune λ
    'smooth_cvar': SmoothCVaR(alpha=0.9, smoothing=0.1),  # Huber-like
    'variance': Variance(),  # Baseline
}

# Only move to CVaR-95/99 after G2 passes
if gate_g2_passed:
    objectives['cvar_95'] = CVaR(0.95)
    objectives['cvar_99'] = CVaR(0.99)
```

### 2.3 Comprehensive Metrics Suite
```python
def compute_metrics(pnl, hedges, payoffs, costs):
    """Critical metrics for gates"""
    unhedged_var = np.var(payoffs)
    hedged_var = np.var(pnl)

    metrics = {
        # Primary
        'hedge_effectiveness': 1 - hedged_var/unhedged_var,
        'cost_to_pnl_ratio': costs.sum() / abs(pnl.sum()),
        'turnover': compute_turnover(hedges),
        'trades_per_day': count_trades(hedges) / n_days,

        # Risk measures
        'sharpe': sharpe_ratio(pnl),
        'sortino': sortino_ratio(pnl),
        'cvar_95': cvar(pnl, 0.95),
        'max_drawdown': max_dd(pnl),

        # vs Baselines
        'vs_bs_sharpe': sharpe(pnl) - sharpe(bs_pnl),
        'vs_unhedged_var': 1 - hedged_var/unhedged_var,
    }
    return metrics
```

### 2.4 Cost Sensitivity Sweep
```python
# Use BEST agent from Phase 2.1-2.3 (already trained!)
best_agent = select_best_from_phase_2()  # LSTM, GRU, or MLP

# Evaluate SAME agent at multiple transaction costs
cost_levels = [0.0005, 0.001, 0.0025, 0.005]  # 5, 10, 25, 50 bps
results = []

for eval_cost in cost_levels:
    # NO TRAINING - just evaluate at different cost
    metrics = evaluate(
        agent=best_agent,              # Same frozen model
        transaction_cost=eval_cost,     # Different cost
        n_paths=10000
    )

    results.append({
        'cost_bps': eval_cost * 10000,
        'trades_per_day': metrics['trades_per_day'],
        'turnover': metrics['turnover'],
        'sharpe': metrics['sharpe'],
    })

# Check monotonicity (sanity check)
for i in range(len(results) - 1):
    current = results[i]
    next_result = results[i+1]

    assert current['trades_per_day'] >= next_result['trades_per_day'], \
        f"Trading should decrease with cost! {current['cost_bps']} bps vs {next_result['cost_bps']} bps"

    assert current['turnover'] >= next_result['turnover'], \
        f"Turnover should decrease with cost!"

# Check graceful degradation
sharpe_degradation = (results[0]['sharpe'] - results[-1]['sharpe']) / results[0]['sharpe']
assert sharpe_degradation < 0.8, f"Performance degrades too much: {sharpe_degradation:.1%}"

print("✓ Cost sensitivity sweep passed")
print(pd.DataFrame(results))
```

**Purpose:** Verify the trained model is robust to transaction cost variation (doesn't overfit to training cost).

**Gate G2: Cannot proceed without:**
- ✓ Mean improvement with 95% CI not crossing 0 (vs unhedged AND BS)
- ✓ Hedge effectiveness > 0.3 (30% variance reduction)
- ✓ Cost-to-PnL ratio < 0.5
- ✓ Consistent across 3+ seeds
- ✓ Cost sensitivity shows monotonic behavior

---

## Phase 3: Objective Function Search (Days 9-11)

### What This Really Is
This is **loss function hyperparameter tuning** - testing which training objective optimizes your actual business metric.

**The Problem:** You're currently training with CVaR-50, but:
- Is that the RIGHT loss function for your goals?
- Does minimizing CVaR-50 actually maximize the metrics you care about (Sharpe? CVaR-95? Hedge effectiveness)?
- Would entropic risk or variance work better?

**The Risk:** Training with the wrong objective could leave 20-30% performance on the table.

### 3.1 Business Objective Documentation
Document what you actually care about:
- **Primary metric**: CVaR-95 (tail risk)? Sharpe (risk-adjusted returns)? Hedge effectiveness?
- **Constraints**: Max trades/day, turnover limits, cost ratios
- **Secondary metrics**: Track these but don't optimize for them

### 3.2 Loss Function Search
Test 3-4 training objectives to find which optimizes your primary metric:
- Entropic risk (λ tuned) - stable baseline
- CVaR-90 - moderate tail risk
- CVaR-95 - strong tail risk
- Variance - simplest

For each loss function:
1. Train with best architecture from Phase 2
2. Evaluate on ALL metrics (especially primary)
3. Compare which training objective produces best primary metric

### 3.3 Selection
Select training objective that produces best results on your primary evaluation metric (NOT based on training loss).

**Note:** This is essentially hyperparameter tuning for the loss function. You could merge this into Phase 2 (comprehensive search) or skip if confident in current objective.

**Gate G3: Cannot proceed without:**
- ✓ Primary business metric documented
- ✓ Training objective selected based on eval results (not train loss)
- ✓ Alignment verified (chosen objective improves primary metric)
- ✓ Business constraints documented

---

## Phase 4: Incremental Scaling (Days 12-17)

### Objectives
Scale up training ONLY after passing G1-G3. Monitor for performance degradation at each step.

### 4.1 Gradual Scale-Up Protocol
Increase training data in stages:
1. **100k paths, 150 epochs** - First scale test
2. **150k paths, 200 epochs** - Mid-scale verification
3. **200k paths, 250 epochs** - Final production scale

**Critical Check:** At each stage, verify performance improves or stays flat (does NOT degrade).

Use best configuration from Phase 2 (architecture, features) and Phase 3 (loss function).

### 4.2 Degradation Detection
Monitor for scaling issues:
- Performance should not drop > 10% from previous scale
- Training curves should remain healthy (smooth, converged, no collapses)
- Overfitting should stay controlled (train/test gap < 30%)

If degradation detected, diagnose root cause (learning rate, batch size, overfitting) before proceeding.

### 4.3 Final Production Model
After completing all scales successfully:
- Save best model with full metadata (config, git hash, metrics)
- Document final hyperparameters
- Checkpoint for Phase 5 validation

**Gate G4: Cannot proceed without:**
- ✓ No degradation at scale (< 10% performance drop)
- ✓ Training curves healthy (smooth, converged)
- ✓ Beats ALL baselines on primary metrics
- ✓ Overfitting under control (train/test gap < 30%)

---

## Phase 5: Multi-Period Robustness Testing (Days 18-20) [OPTIONAL]

### Reality Check: Synthetic Training Data Limitation

**Current Setup:** Your implementation trains on **synthetic GBM data** (not historical data), so traditional walk-forward validation (train on historical period 1, test on historical period 2) doesn't apply.

**What You Already Have:** Bootstrap backtesting via `--n-bootstrap-paths` parameter in `/Users/shaobaolin/Documents/repo/pfhedge/crypto/backtest/run.py`. This tests the trained model on multiple synthetic market scenarios.

**What Phase 5 Adds:** Additional robustness testing beyond bootstrap, using your existing infrastructure.

### 5.1 Bootstrap Robustness Suite (Existing Infrastructure)

Use your existing `--n-bootstrap-paths` to test the trained model on many synthetic scenarios:

```bash
# Test on 100 different synthetic market paths
python -m crypto.backtest.run \
    --config backtest.yaml \
    --n-bootstrap-paths 100 \
    --seed 42
```

**Analyze:**
- Distribution of Sharpe ratios across bootstrap samples
- Win rate vs. baselines (what % of samples beat BS delta?)
- Stability metrics (std of hedge effectiveness)
- Worst-case analysis (CVaR-95 of PnL distribution)

**Target:** Model beats baseline in > 70% of bootstrap samples

### 5.2 Parameter Robustness Testing (2-3 hours)

Test trained model's sensitivity to simulator parameters:

```python
# Test same trained model on different market conditions
test_configs = {
    'base_vol': {'sigma': 0.8},           # Original training vol
    'low_vol': {'sigma': 0.5},            # Calmer markets
    'high_vol': {'sigma': 1.2},           # Volatile markets
    'low_cost': {'transaction_cost': 0.03%},
    'high_cost': {'transaction_cost': 0.10%},
}

for name, config in test_configs.items():
    metrics = evaluate(trained_model, **config, n_paths=10000)
    # Check graceful degradation
```

**Target:** Performance degrades < 40% in stress scenarios

### 5.3 Multiple Time Horizons (1-2 hours)

Test model on options with different maturities than training:

```python
# Trained on 42-day options (original setup)
# Test on different maturities
test_maturities = [21, 42, 63, 84]  # 3 weeks, 6 weeks, 12 weeks

for days in test_maturities:
    metrics = evaluate_maturity(trained_model, days_to_expiry=days)
```

**Target:** Model generalizes reasonably to ±50% maturity range

### 5.4 Seed Robustness (Already Tested in Phase 2)

Phase 2 already tests across 3+ seeds. If not done comprehensively:
- Train 5-10 models with different seeds
- Measure variance of performance metrics
- Document best/worst/median cases

**Target:** Coefficient of variation < 0.3 for primary metrics

### 5.5 Simplified Validation Report

Deliverables:
- Bootstrap distribution analysis (100+ samples)
- Parameter robustness results (vol/cost stress tests)
- Maturity generalization check
- Seed stability summary (from Phase 2)
- Decision: Is model robust enough for production?

**Gate G5 (PRODUCTION): Cannot proceed without:**
- ✓ Beats baseline in > 70% of bootstrap samples
- ✓ Graceful degradation in stress scenarios (< 40% drop)
- ✓ Generalizes to ±50% maturity range
- ✓ Low variance across seeds (CV < 0.3)
- ✓ No catastrophic failures in any scenario

### Why This Is Different from Phase 2

**Phase 2** = Find best architecture/loss function on standard training config
**Phase 5** = Stress-test chosen model on diverse conditions it hasn't seen

### When Walk-Forward WILL Matter (Future Work)

Walk-forward validation becomes critical when you:
1. Train on **historical data** (not synthetic GBM)
2. Implement **GAN-based simulator** (trains on real market data)
3. Deploy to **production** (periodic retraining on new data)

At that point, traditional walk-forward with train/test periods on historical dates will be essential. For now, bootstrap + stress testing is the appropriate validation approach.

---

## Engineering Best Practices Checklist

### Training Configuration
```python
best_practices = {
    'mixed_precision': True,
    'gradient_clipping': 1.0,
    'dropout': 0.2,
    'recurrent_dropout': 0.1,
    'layer_norm': True,
    'batch_norm': False,  # Often unstable with RNNs
    'optimizer': 'AdamW',
    'weight_decay': 1e-4,
    'lr_scheduler': 'ReduceLROnPlateau',
    'early_stopping': True,
    'checkpoint_best': True,
    'seed_everything': True,
}
```

### Online Feature Construction
```python
class OnlineFeatureConstructor:
    """Ensure same feature construction in train/test"""

    def __init__(self, lookback_window=20):
        self.window = lookback_window

    def construct_features(self, data, t):
        """Strictly past-only construction"""
        assert t >= self.window, "Not enough history"

        # Only use data[:t] (not including t)
        past_data = data[:t]

        features = {
            'log_moneyness': np.log(past_data[-1] / strike),
            'time_to_expiry': (T - t) / T,
            'realized_vol': np.std(np.diff(np.log(past_data[-self.window:]))),
            # NO forward-looking features
        }

        return features
```

---

## Daily Execution Template

```markdown
## Day X: [Phase.Task]

### Morning
- [ ] Review previous day results & gate status
- [ ] Check background jobs status
- [ ] Plan experiments for today

### Execution
- [ ] Run experiment with seeds [42, 43, 44]
- [ ] Log to backtest_results/{timestamp}/
- [ ] Monitor training curves in real-time
- [ ] Check for anomalies (gradient explosion, etc.)

### Analysis
- [ ] Compute all metrics (not just Sharpe)
- [ ] Compare to ALL baselines
- [ ] Statistical significance test (CI)
- [ ] Cost sensitivity check

### Documentation
- [ ] Update results spreadsheet
- [ ] Git commit with experiment tag
- [ ] Document any issues/blockers
- [ ] Write decision for tomorrow

### Gate Check
- [ ] Review gate criteria
- [ ] Document pass/fail status
- [ ] If fail: diagnose root cause
```

---

## Risk Registry & Mitigations

| Risk | Detection | Mitigation |
|------|-----------|------------|
| Simulator drift | Phase 1 t-test | Risk-neutralization |
| Feature leakage | Unit test | Online construction |
| Objective misalignment | Phase 3 testing | Align train/eval |
| Overfitting | Train/test gap > 30% | Early stopping, dropout |
| Cost instability | Sensitivity sweep | Test multiple levels |
| Parameter drift | WF stability metric | Regularization |
| Compute explosion | Wall time limits | Auto early-stop |

---

## Success Criteria Summary

### Phase Gates (Must Pass to Proceed)

| Gate | Criteria | Measurement |
|------|----------|-------------|
| G0 | Infrastructure ready | All tools working |
| G1 | Simulator verified | GBM sanity check passes |
| G2 | Architecture selected | 95% CI improvement |
| G3 | Objective aligned | Train→eval correlation |
| G4 | Scaling successful | No degradation |
| G5 | Production ready | Bootstrap win rate > 70%, stress tests pass |

### Final Targets (Day 20)

| Metric | Minimum | Target |
|--------|---------|--------|
| Bootstrap win rate | > 70% | > 80% |
| Hedge effectiveness | > 40% | > 50% |
| Cost-to-PnL | < 50% | < 30% |
| Seed stability (CV) | < 0.4 | < 0.3 |
| Stress degradation | < 50% | < 40% |
| Sharpe vs BS | > 0 | > 1.0 |

---

## Timeline

| Phase | Days | Key Output |
|-------|------|------------|
| Phase 0 | 1-2 | Infrastructure ready |
| **Phase 1** | **3** | **GBM sanity check (1 hour)** |
| Phase 2 | 4-8 | Best architecture |
| Phase 3 | 9-11 | Aligned objective |
| Phase 4 | 12-17 | Scaled model |
| Phase 5 | 18-20 | Robustness validation (optional) |

**Total: 20 days to production-ready system** (down from 28!)

---

## Conclusion

This revised plan addresses all critical feedback:
- **Martingale verification** uses proper statistical testing (critical for future GAN work)
- **Pure hedging test** has meaningful criteria
- **Objective selection** starts with stable entropic
- **Metrics** are comprehensive, not Sharpe-only
- **Bootstrap validation** replaces walk-forward (appropriate for synthetic training data)
- **Stress testing** checks robustness to parameter variation
- **Cost sensitivity** includes sweep and monotonicity
- **Engineering guardrails** prevent common failures
- **Feature leakage** prevented with online construction

The plan systematically de-risks the project through explicit gates, comprehensive measurement, and rigorous validation before production deployment.

**Key Insight:** Phase 5 now reflects the reality that you train on synthetic GBM data, not historical data. Bootstrap + stress testing is the appropriate validation approach until you implement GAN-based simulators (future work).

---

## Appendix: Future Work - GAN-Based Market Simulator

**When to implement:** After Phase 5 is complete and production system is stable

### Why GAN Simulators Matter

Current GBM simulator limitations:
- Constant volatility (no volatility clustering)
- No jumps or fat tails
- No correlation structure with other assets
- Misses stylized facts of crypto markets

GAN-based simulators can learn these patterns directly from data, producing more realistic training environments.

### Implementation Phases for GAN

**Phase GAN-1: Drift Detection & Risk-Neutralization (5 days)**

This is when the original Phase 1 tests become CRITICAL:

```python
### GAN-1.1 Martingale Property Test
def verify_martingale_property_gan(gan_simulator, n_paths=50_000):
    """GAN outputs may have learned drift from training data"""
    paths = gan_simulator.generate(n_paths)

    # Statistical t-test (not fixed threshold)
    S0 = paths[:, 0]
    ST = paths[:, -1]
    mean_return = (ST / S0).mean()
    std_error = (ST / S0).std() / np.sqrt(n_paths)
    t_stat = (mean_return - 1.0) / std_error

    # GAN likely fails this test initially
    if abs(t_stat) > 3:
        print(f"⚠ GAN has drift: E[ST/S0] = {mean_return:.4f}")
        return False
    return True

### GAN-1.2 Pure Hedging Test
# Train agent on GAN data - will it learn to exploit drift?
agent = MLPHedger()
agent.fit(gan_simulator, n_paths=50_000)

# If mean hedge position is biased, GAN has drift
hedge_positions = agent.compute_hedge(test_scenarios)
if abs(hedge_positions.mean()) > 0.2:
    print("⚠ Agent learned to exploit GAN drift")

### GAN-1.3 Risk-Neutralization
# Apply Buehler et al. method to remove learned drift
gan_risk_neutral = find_risk_neutral_measure(gan_simulator)
# Re-verify martingale property
assert verify_martingale_property_gan(gan_risk_neutral)
```

**Phase GAN-2: GAN Architecture Selection (2 weeks)**
- WGAN-GP with LSTM generator/discriminator
- TimeGAN for temporal dynamics
- Compare sample quality (Wasserstein distance, MMD)

**Phase GAN-3: Training Data Preparation (1 week)**
- Collect 2+ years of high-frequency BTC data
- Feature engineering (returns, volatility, volume)
- Train/validation split

**Phase GAN-4: GAN Training & Validation (2 weeks)**
- Train GAN on historical data
- Validate generated paths match stylized facts
- Apply risk-neutralization
- Re-run all Phase 2-5 tests with GAN simulator

**Expected Benefits:**
- 20-40% improvement in hedge effectiveness
- Better generalization to unseen market regimes
- Capture volatility clustering and jumps

**Risk:** More complex simulator = more model risk. Only implement after baseline GBM system is production-ready and validated.