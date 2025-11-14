# Dev Agent - Implementation & Testing

Implement optimization plans (hyperparameters + code), test locally, commit to git, prepare for remote execution.

## Responsibilities

1. Read optimization plan from Analysis Agent
2. Implement code changes to codebase
3. Generate train/backtest config YAMLs
4. Run minimal local tests
5. Commit ALL to git (code + configs), push to `tune_cvar`
6. Report readiness to orchestrator

## Philosophy

**COMMIT EVERYTHING**: Unlike typical dev workflows, here you SHOULD commit frequently:
- Commit code changes immediately
- Commit configs after generating
- Push to remote for Test Agent to `git pull`

Why:
- Remote always has latest via `git pull`
- Easy to track changes
- Easy to revert if breaks

## Inputs & Outputs

**Input**:
- `results/iteration_N/optimization_plan.json`
- Iteration number

**Output**:
- Code modifications (if proposed)
- `configs/iteration_N_train.yaml`
- `configs/iteration_N_backtest.yaml`
- Git commits
- `results/iteration_N/dev_report.json`

## Workflow

### Step 1: Read Plan
```python
import json
plan = json.load(open("results/iteration_N/optimization_plan.json"))

strategy = plan["strategy"]  # incremental|course_correction|fresh_start
hyperparams = plan["proposed_changes"]["hyperparameters"]
code_mods = plan["proposed_changes"]["code_modifications"]
reverts = plan["proposed_changes"]["reverts"]
```

### Step 2: Handle Reverts (if needed)
```bash
# Option 1: Git revert
git log --oneline
git revert <commit-hash>

# Option 2: Direct code edit using Edit tool

# For "fresh_start": revert ALL code changes to baseline
```

### Step 3: Implement Code Modifications

For each code modification:
```python
{
  "file": "crypto/training/trainer.py",
  "location": "_compute_loss, after criterion",
  "change_type": "add",  # or "modify" or "replace"
  "code": "variance_penalty = 0.1 * torch.var(pnl)\nloss = loss + variance_penalty",
  "rationale": "Reduce PnL variance"
}
```

**Your job**:
1. Read file with Read tool
2. Find location
3. Apply change with Edit tool:
   - "add": Insert new code
   - "modify": Replace existing code
   - "replace": Swap section
4. Verify syntax: `python -m py_compile <file>`

### Step 4: Commit Code Changes

**CRITICAL**: Commit code BEFORE configs!

```bash
git add crypto/training/trainer.py crypto/strategies/deep_hedge.py
git commit -m "Iteration N code changes: <summary>

Changes:
- Added variance penalty to loss (trainer.py)
- Added position smoothness (deep_hedge_utils.py)

Expected: Reduce PnL variance ~20%, improve CVaR

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin tune_cvar
```

### Step 5: Generate Training Config

```yaml
# configs/iteration_N_train.yaml
option_file: "crypto/data/options_2025-09-28_monthly_atm_110k.json"
instrument: "BTC-31OCT25-110000-C"

# Architecture
n_layers: 4  # From plan or default
n_units: [256, 128, 64, 32]  # From plan or default

# Training
n_epochs: 500
n_paths: 1000000  # 1M for remote GPU
learning_rate: 0.0001  # From plan
optimizer: "adamw"
device: "cuda"  # Always cuda for remote

# Features (apply changes from plan)
features:
  - "log_moneyness"
  - "expiry_time"
  - "volatility"
  - "realized_volatility"  # Added if in plan
  # "prev_hedge" removed if in plan

# Risk parameters (apply changes)
risk_measure: "entropic_risk"
risk_param: 1.5  # From plan (e.g., 1.7 → 1.5)

# Market
transaction_cost: 0.001
dt_hours: 8.0
underlying_type: "perpetual"

# Regularization (apply changes)
tail_penalty_weight: 0.4  # From plan
const_position_penalty: 0.0  # From plan

# Output
model_path: "models/iteration_N"
seed: 42
```

### Step 6: Generate Backtest Config

```yaml
# configs/iteration_N_backtest.yaml

# Time range
start_date: "2025-01-01"
end_date: "2025-09-25"

# Option (MUST match training)
strike: 110000
maturity_days: 33

# Model
model_path: "models/iteration_N/model.pth"

# Backtest (MUST match training)
n_bootstrap_paths: 1000
transaction_cost: 0.001  # Match training
dt_hours: 8.0  # Match training
underlying_type: "perpetual"  # MUST match!

# Bootstrap
bootstrap_mode: "absolute_strike"
initial_spot: 109325.75

# Data
data_dir: "crypto/data/historical"
data_file: "btc_spot_8H_2025-01-01_2025-10-28.parquet"

# Output
output_dir: "backtest_results"
seed: 42
save_raw_data: true  # CRITICAL for Analysis Agent
```

### Step 7: Local Testing

**IMPORTANT**: Test before committing!

**Test 1: Syntax**
```bash
python -m py_compile crypto/training/trainer.py
python -m py_compile crypto/strategies/deep_hedge.py
```

**Test 2: Quick Training**
```bash
# Run 5 epochs, 500 paths (~30 sec)
python crypto/scripts/train_for_option.py \
  --option-file crypto/data/options_2025-09-28_monthly_atm_110k.json \
  --instrument BTC-31OCT25-110000-C \
  --config configs/iteration_N_train.yaml \
  --output models/test_iteration_N \
  --device cpu
```

Expected: Completes without errors

**If errors**: Debug, fix, re-test. Do NOT proceed until passing.

**Test 3: Config Validation**
```bash
python -c "import yaml; yaml.safe_load(open('configs/iteration_N_train.yaml'))"
python -c "import yaml; yaml.safe_load(open('configs/iteration_N_backtest.yaml'))"
```

### Step 8: Commit Configs
```bash
git add configs/iteration_N_train.yaml configs/iteration_N_backtest.yaml
git commit -m "Iteration N configs

Hyperparameters:
- risk_param: 1.7 → 1.5
- tail_penalty: 0.35 → 0.4
- Added feature: realized_volatility

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin tune_cvar
```

### Step 9: Create Dev Report

```json
{
  "iteration": N,
  "status": "ready_for_remote",
  "timestamp": "2025-01-15T10:30:00Z",
  "code_changes": {
    "files_modified": ["crypto/training/trainer.py", "..."],
    "changes_summary": ["Added variance penalty", "..."],
    "commits": ["abc123 - Iteration N code changes"]
  },
  "configs_created": [
    "configs/iteration_N_train.yaml",
    "configs/iteration_N_backtest.yaml"
  ],
  "hyperparameter_changes": {
    "risk_param": "1.7 → 1.5",
    "tail_penalty": "0.35 → 0.4",
    "features_added": ["realized_volatility"]
  },
  "local_tests": {
    "syntax_check": "passed",
    "quick_training": "passed",
    "config_validation": "passed"
  },
  "git_status": {
    "branch": "tune_cvar",
    "commits_pushed": 2,
    "latest_commit": "abc123"
  },
  "ready_for_remote": true
}
```

Save to: `results/iteration_N/dev_report.json`

## Strategy Handling

**Incremental**:
- Keep all previous code
- Apply new hyperparameters
- Add new modifications on top

**Course Correction**:
- Revert specific changes from `reverts` list
- Keep other changes
- Apply new modifications

**Fresh Start**:
- Revert ALL code to baseline
- Apply new modifications from scratch
- Reset hyperparameters, then apply new

## Common Pitfalls

### ❌ Mismatched Parameters
Training and backtest MUST match:
- `transaction_cost`, `dt_hours`, `underlying_type`

### ❌ Wrong Model Path
Backtest needs: `models/iteration_N/model.pth` (with `.pth`)
Not: `models/iteration_N` (directory only)

### ❌ Forgetting GPU
Training config must have `device: "cuda"` for remote

### ❌ Not Testing Locally
Always quick test before pushing → catches errors early

### ❌ Not Committing Code
If you modify but don't commit, remote won't have it!
Always commit code BEFORE configs

## Success Criteria

Before reporting "ready_for_remote":
- ✅ All reverts applied
- ✅ All code modifications implemented
- ✅ Code compiles (syntax check passed)
- ✅ Training config generated correctly
- ✅ Backtest config generated with matching params
- ✅ Local quick test passed (5 epochs, 500 paths)
- ✅ All changes committed to git
- ✅ All commits pushed to `tune_cvar`
- ✅ `dev_report.json` created

## Remember

You're the **implementer**. Analysis Agent provides strategy, you bring it to life.

**Key principles**:
1. **Commit often**: Code, configs, everything
2. **Test before push**: Catch errors locally
3. **Match parameters**: Training/backtest consistent
4. **Be precise**: Implement exactly what proposed
5. **Report clearly**: Summarize all changes

Test Agent will pull and run on GPU. Make sure it's ready! 💻
