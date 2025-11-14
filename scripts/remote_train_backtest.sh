#!/bin/bash
set -e

# Remote Training & Backtesting Script
# This script runs on the remote GPU server at /workspace/pfhedge
# Usage: bash remote_train_backtest.sh <iteration_number> <instrument_name>

# Initialize conda
export PATH="/opt/miniforge3/bin:$PATH"

ITERATION=$1
INSTRUMENT=$2

if [ -z "$ITERATION" ] || [ -z "$INSTRUMENT" ]; then
    echo "Error: Missing arguments"
    echo "Usage: bash remote_train_backtest.sh <iteration_number> <instrument_name>"
    echo "Example: bash remote_train_backtest.sh 1 BTC-31OCT25-110000-C"
    exit 1
fi

echo "========================================="
echo "Iteration $ITERATION - Remote Execution"
echo "Instrument: $INSTRUMENT"
echo "Start time: $(date)"
echo "========================================="

# Navigate to project directory
cd /workspace/pfhedge

# Pull latest code from tune_cvar branch
echo ""
echo "[1/4] Pulling latest code from git..."
git reset --hard HEAD
git clean -fd -e 'configs/iteration_*'
git pull origin tune_cvar

# Verify configs exist
if [ ! -f "configs/iteration_${ITERATION}_train.yaml" ]; then
    echo "Error: Training config not found: configs/iteration_${ITERATION}_train.yaml"
    exit 1
fi

if [ ! -f "configs/iteration_${ITERATION}_backtest.yaml" ]; then
    echo "Error: Backtest config not found: configs/iteration_${ITERATION}_backtest.yaml"
    exit 1
fi

echo "✓ Configs found"

# Training
echo ""
echo "========================================="
echo "[2/4] Training iteration $ITERATION"
echo "========================================="
echo "Start training: $(date)"

/venv/main/bin/python crypto/scripts/train_for_option.py \
  --option-file crypto/data/options_2025-09-28_monthly_atm_110k.json \
  --instrument "$INSTRUMENT" \
  --config configs/iteration_${ITERATION}_train.yaml \
  --output models/iteration_${ITERATION}

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "Error: Training failed with exit code $TRAINING_EXIT_CODE"
    exit $TRAINING_EXIT_CODE
fi

echo "✓ Training complete: $(date)"

# Find actual model directory (may have git hash appended) - use newest
MODEL_DIR=$(ls -td models/iteration_${ITERATION}* 2>/dev/null | head -1)

if [ -z "$MODEL_DIR" ]; then
    echo "Error: No model directory found matching: models/iteration_${ITERATION}*"
    exit 1
fi

echo "Found model directory: $MODEL_DIR"

# Verify model was created
if [ ! -f "$MODEL_DIR/model.pth" ]; then
    echo "Error: Model file not created: $MODEL_DIR/model.pth"
    exit 1
fi

if [ ! -f "$MODEL_DIR/training_results.json" ]; then
    echo "Error: Training results not created: $MODEL_DIR/training_results.json"
    exit 1
fi

echo "✓ Model and training results verified"

# Backtesting
echo ""
echo "========================================="
echo "[3/4] Backtesting iteration $ITERATION"
echo "========================================="
echo "Start backtest: $(date)"

# Update backtest config with actual model path (use absolute path)
ABSOLUTE_MODEL_PATH="/workspace/pfhedge/$MODEL_DIR/model.pth"
ABSOLUTE_DATA_DIR="/workspace/pfhedge/crypto/data/historical"
ABSOLUTE_OUTPUT_DIR="/workspace/pfhedge/backtest_results"

sed -i "s|model_path:.*|model_path: \"$ABSOLUTE_MODEL_PATH\"|" configs/iteration_${ITERATION}_backtest.yaml
sed -i "s|data_dir:.*|data_dir: \"$ABSOLUTE_DATA_DIR\"|" configs/iteration_${ITERATION}_backtest.yaml
sed -i "s|output_dir:.*|output_dir: \"$ABSOLUTE_OUTPUT_DIR\"|" configs/iteration_${ITERATION}_backtest.yaml

echo "Updated backtest config:"
echo "  Model: $ABSOLUTE_MODEL_PATH"
echo "  Data: $ABSOLUTE_DATA_DIR"
echo "  Output: $ABSOLUTE_OUTPUT_DIR"

/venv/main/bin/python -m crypto.backtest.run \
  --config configs/iteration_${ITERATION}_backtest.yaml \
  --seed 42

BACKTEST_EXIT_CODE=$?

if [ $BACKTEST_EXIT_CODE -ne 0 ]; then
    echo "Error: Backtesting failed with exit code $BACKTEST_EXIT_CODE"
    exit $BACKTEST_EXIT_CODE
fi

echo "✓ Backtesting complete: $(date)"

# Verify results were created
if [ ! -f "backtest_results/results.json" ]; then
    echo "Error: Backtest results not created: backtest_results/results.json"
    exit 1
fi

echo "✓ Backtest results verified"

# Summary
echo ""
echo "========================================="
echo "[4/4] Iteration $ITERATION - Summary"
echo "========================================="

# Extract key metrics from training results
echo ""
echo "Training Metrics:"
/venv/main/bin/python -c "
import json
try:
    with open('$MODEL_DIR/training_results.json', 'r') as f:
        results = json.load(f)
    print(f\"  Initial Loss: {results['summary']['initial_loss']:.2f}\")
    print(f\"  Final Loss: {results['summary']['final_loss']:.2f}\")
    print(f\"  Improvement: {results['summary']['improvement_pct']:.1f}%\")
except Exception as e:
    print(f\"  Error reading training results: {e}\")
"

# Extract key metrics from backtest results
echo ""
echo "Backtest Metrics:"
/venv/main/bin/python -c "
import json
try:
    with open('backtest_results/results.json', 'r') as f:
        results = json.load(f)
    deep = results['summary']['deep_hedge']
    bs = results['summary']['bs_baseline']
    print(f\"  Deep Hedge:\")
    print(f\"    Sharpe Ratio: {deep['sharpe_ratio']:.2f}\")
    print(f\"    CVaR 95%: {deep['cvar_95']:.2f}\")
    print(f\"    Mean PnL: {deep['mean']:.2f}\")
    print(f\"    Std PnL: {deep['std']:.2f}\")
    print(f\"  BS Baseline:\")
    print(f\"    Sharpe Ratio: {bs['sharpe_ratio']:.2f}\")
    print(f\"    CVaR 95%: {bs['cvar_95']:.2f}\")
except Exception as e:
    print(f\"  Error reading backtest results: {e}\")
"

echo ""
echo "========================================="
echo "Iteration $ITERATION complete!"
echo "End time: $(date)"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Download results from local machine using trigger_remote.sh"
echo "  2. Or manually download:"
echo "     scp -P 47612 root@185.65.93.114:/workspace/pfhedge/backtest_results/results.json ."
echo "     scp -P 47612 root@185.65.93.114:/workspace/pfhedge/models/iteration_${ITERATION}/training_results.json ."
echo ""
