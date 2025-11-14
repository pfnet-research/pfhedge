# Iteration 18: Performance Improvement Plan

**Date:** 2025-01-11
**Base Model:** Iteration 16 (Best model so far)
**Strategy:** Phase 1 - Quick wins with conservative, proven changes

---

## Current Performance (Iteration 16 Baseline)

| Metric | Deep Hedge | BS Baseline | vs BS |
|--------|-----------|-------------|-------|
| Mean PnL | -$3,957 | -$4,259 | **+$302** ✅ |
| CVaR (95%) | -$12,033 | -$12,669 | **+$636** ✅ |
| Sharpe Ratio | -1.097 | -1.065 | -0.032 ⚠️ |
| Win Rate | 8.2% | - | - |

**Critical Success:** Actually hedging properly (unlike iteration 17)

### Current Configuration
```yaml
# Architecture
n_units: [256, 128, 64, 32]

# Features (4 total)
features:
  - log_moneyness
  - expiry_time
  - volatility  # 20-period rolling
  - prev_hedge

# Training
tail_penalty_weight: 0.5          # CRITICAL - maintains hedging
tail_penalty_ramp: false          # CRITICAL - never enable!
band_width: 0.001
transaction_cost: 0.0004
learning_rate: 1e-3
weight_decay: 1e-4

# Option
strike: 110000
maturity_days: 33
call: true
dt_hours: 8.0
```

---

## Phase 1: Quick Wins Implementation

### Goal
Improve iteration 16 performance by 10-15% across all metrics using only proven, low-risk changes while maintaining proper hedging behavior.

### Expected Improvements
- **Mean PnL:** -$3,957 → -$3,400 (+$557, 14% improvement)
- **CVaR:** -$12,033 → -$10,200 (+$1,833, 15% improvement)
- **Sharpe:** -1.097 → -0.95 (+13% improvement)
- **Hedging:** Maintain 0.4-0.6 range for ATM calls

---

## Changes to Implement

### 1. Enable Black-Scholes Warmup Training ✅ HIGH PRIORITY

**What:** Train first 15 epochs to imitate BS delta, then gradually transition to CVaR optimization

**Why:**
- Better initialization ensures hedging behavior from start
- Reduces risk of model learning non-hedging strategies
- Proven technique in deep hedging literature

**Implementation:**
```yaml
# Add to training config
bs_warmup_epochs: 15              # NEW: Imitate BS for first 15 epochs
curriculum_ramp_epochs: 10        # NEW: Gradually transition to CVaR
bs_anchor_weight: 0.1             # NEW: Keep 10% BS similarity throughout
```

**Code Location:** Already implemented in `crypto/training/trainer.py` - just needs config activation

**Risk:** LOW - Safe initialization technique
**Complexity:** SIMPLE (config change only)
**Expected Impact:** CVaR +5-10%, better training stability

---

### 2. Add Multi-Window Volatility Features ✅ HIGH PRIORITY

**What:** Add short-term (10-period) and long-term (50-period) volatility alongside existing 20-period

**Why:**
- Captures volatility regime changes more effectively
- Short-term vol detects spikes and tail events
- Long-term vol provides stable trend information
- Proven effective in crypto markets with regime shifts

**Implementation:**
```yaml
# Update feature list from 4 to 6 features
features:
  - log_moneyness
  - expiry_time
  - volatility           # Keep existing 20-period
  - volatility_10        # NEW: Short-term (captures spikes)
  - volatility_50        # NEW: Long-term (captures regime)
  - prev_hedge
```

**Code Changes:**
1. Add multi-window support to `crypto/strategies/deep_hedge_utils.py`
2. Register new features in pfhedge feature system
3. Update instrument volatility calculation

**Risk:** LOW - Just adding more of what already works
**Complexity:** SIMPLE (feature engineering)
**Expected Impact:** CVaR +2-5%, Mean PnL +$100-300

---

### 3. Enable Learning Rate Scheduler ✅ MEDIUM PRIORITY

**What:** Use cosine annealing to gradually reduce learning rate during training

**Why:**
- Better convergence to local minima
- Reduces risk of overshooting optimal parameters
- Standard optimization technique

**Implementation:**
```yaml
# Add to training config
use_lr_scheduler: true            # NEW: Enable cosine annealing
```

**Code Location:** Already implemented in trainer - just needs activation

**Risk:** LOW - Standard technique
**Complexity:** SIMPLE (config change)
**Expected Impact:** Mean PnL +$50-100, more stable final loss

---

## Safety Measures (DO NOT CHANGE)

These settings MUST remain unchanged to maintain hedging quality:

```yaml
tail_penalty_weight: 0.5          # ✅ KEEP - Critical for hedging
tail_penalty_ramp: false          # ✅ KEEP - Never enable (causes flat hedging)
band_width: 0.001                 # ✅ KEEP - Working well
transaction_cost: 0.0004          # ✅ KEEP - Realistic cost
```

---

## Success Criteria

**Critical Checks (in order):**

1. **Hedge Positions FIRST** (most important)
   - Mean hedge ratio: 0.4-0.6 for ATM calls ✅
   - Std hedge ratio: 0.1-0.2 (reasonable variation) ✅
   - No negative or near-zero positions ✅

2. **Performance Metrics** (secondary)
   - Mean PnL better than iteration 16 ✅
   - CVaR better than iteration 16 ✅
   - Sharpe ratio improved ✅

3. **Training Stability**
   - Loss converges smoothly ✅
   - No divergence or NaN values ✅
   - Final loss < initial loss ✅

**Never optimize metrics without verifying hedging behavior.**

---

## Implementation Plan

### Files to Create/Modify

1. **`configs/iteration_18_train.yaml`**
   - Based on iteration_16_train.yaml
   - Add: bs_warmup_epochs, curriculum_ramp_epochs, bs_anchor_weight
   - Add: use_lr_scheduler
   - Update: features list (add volatility_10, volatility_50)

2. **`crypto/strategies/deep_hedge_utils.py`**
   - Add multi-window volatility feature support
   - Register volatility_10 and volatility_50 features

3. **`configs/iteration_18_backtest.yaml`**
   - Based on iteration_16_backtest.yaml
   - Update: model_path to iteration_18 model
   - Update: features list to match training

### Timeline

- **Config creation:** 15 min
- **Feature implementation:** 30-45 min
- **Local testing:** 15 min (5 epochs, 1k paths)
- **Training (remote GPU):** ~2 hours (100 epochs, 10k paths)
- **Backtesting:** ~15 min (1k bootstrap paths)
- **Analysis:** 30 min
- **Total:** ~3-4 hours

---

## Future Phases (NOT in Iteration 18)

### Phase 2: Feature Expansion (Future)
- Add volatility skew (vol_short/vol_long ratio)
- Add spot momentum (trend indicator)
- Add distance-to-strike (magnitude of moneyness)
- Expected: +10-20% additional improvement

### Phase 3: Advanced Training (Future)
- Premium-aware loss function
- Multi-seed ensemble (5 models)
- Residual connections in architecture
- Expected: +15-25% additional improvement

### Phase 4: Production Readiness (Future)
- Multi-regime validation
- Out-of-sample testing
- Funding rate integration (crypto-specific)

---

## Risk Assessment

### Low Risk (Safe to Implement) ✅
- BS warmup training
- Multi-window volatility
- LR scheduler
- All Phase 1 changes

### Medium Risk (Phase 2+)
- Volatility skew
- Spot momentum
- Premium-aware loss
- Larger architecture

### High Risk (Phase 3+)
- Attention mechanisms
- Funding rate integration
- Very aggressive architectures

---

## Key Learnings from Previous Iterations

1. **Iteration 17 Failure:** CVaR optimization with tighter bands
   - Caused flat/near-zero hedging
   - Good metrics but NOT actually hedging
   - **Lesson:** Always check hedge positions FIRST

2. **Iteration 16 Success:** Conservative approach
   - tail_penalty_weight: 0.5 (not ramped)
   - Simple features, proven architecture
   - Actually hedges properly
   - **Lesson:** Start conservative, validate, then expand

3. **General Principle:**
   - Hedging behavior > Metric optimization
   - Incremental improvements > Big changes
   - Test thoroughly > Rush to production

---

## Validation Protocol

### After Training Completes:

1. **Check Training Metrics**
   ```python
   # Verify loss decreased
   initial_loss = training_results['summary']['initial_loss']
   final_loss = training_results['summary']['final_loss']
   assert final_loss < initial_loss * 0.8  # At least 20% improvement
   ```

2. **Check Backtest Results**
   ```python
   # Load results
   results = json.load(open('results/iteration_18/results.json'))

   # CRITICAL: Check hedge positions FIRST
   hedge_mean = results['detailed']['deep_hedge']['hedge_position_mean']
   hedge_std = results['detailed']['deep_hedge']['hedge_position_std']

   assert 0.4 <= hedge_mean <= 0.6, f"Hedge mean {hedge_mean} out of range!"
   assert 0.1 <= hedge_std <= 0.2, f"Hedge std {hedge_std} out of range!"

   # Then check metrics
   deep_pnl = results['summary']['deep_hedge']['mean']
   base_pnl = -3957  # Iteration 16 baseline
   assert deep_pnl > base_pnl, f"PnL worse than baseline: {deep_pnl}"
   ```

3. **Visual Inspection**
   - Plot hedge positions over time - should vary between 0-1
   - Plot PnL distribution - should be reasonable
   - Check for anomalies (spikes, flat regions)

---

## Iteration 18 Config Template

```yaml
# configs/iteration_18_train.yaml

# Model Architecture
model_type: enhanced_mlp
n_units: [256, 128, 64, 32]  # Keep proven architecture
dropout: 0.1
use_layer_norm: true

# Features (6 total - added multi-window vol)
features:
  - log_moneyness
  - expiry_time
  - volatility          # 20-period (existing)
  - volatility_10       # NEW: Short-term
  - volatility_50       # NEW: Long-term
  - prev_hedge

# Training - BS Warmup (NEW)
n_epochs: 100
n_paths: 10000
bs_warmup_epochs: 15              # NEW
curriculum_ramp_epochs: 10        # NEW
bs_anchor_weight: 0.1             # NEW
use_lr_scheduler: true            # NEW

# Loss Function (KEEP SAFE SETTINGS)
criterion: expected_shortfall
risk_param: 0.9
tail_penalty_weight: 0.5          # CRITICAL - do not change
tail_penalty_ramp: false          # CRITICAL - never enable
variance_penalty_weight: 0.0

# Optimizer
optimizer: adamw
learning_rate: 1e-3
weight_decay: 1e-4
batch_size: null  # Use full batch

# Option Parameters
strike: 110000
maturity_days: 33
call: true
underlying_type: perpetual
dt_hours: 8.0

# Transaction Costs
transaction_cost: 0.0004
band_width: 0.001

# Other
seed: 42
device: cuda
```

---

## Notes

- **Conservative approach:** Only implement proven techniques in Phase 1
- **Validation first:** Check hedging behavior before celebrating metrics
- **Iterative improvement:** Phase 2+ only if Phase 1 succeeds
- **Documentation:** Update this doc with actual results after iteration 18 completes

---

## References

- Iteration 16 Results: `docs/iteration_16/backtest_report.md`
- Iteration 17 Lessons: `docs/CRITICAL_LESSONS_LEARNED.md`
- Training Framework: `crypto/training/USAGE.md`
- Backtest Framework: `crypto/backtest/USAGE.md`
