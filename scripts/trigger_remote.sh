#!/bin/bash
set -e

# Local Trigger Script for Remote Training/Backtesting
# This script runs on your local machine and orchestrates remote execution
# Usage: bash trigger_remote.sh <iteration_number>

ITERATION=$1
INSTRUMENT="BTC-31OCT25-110000-C"
REMOTE_HOST="root@185.65.93.114"
REMOTE_PORT="47612"
REMOTE_PATH="/workspace/pfhedge"

if [ -z "$ITERATION" ]; then
    echo "Error: Missing iteration number"
    echo "Usage: bash trigger_remote.sh <iteration_number>"
    echo "Example: bash trigger_remote.sh 1"
    exit 1
fi

echo "========================================="
echo "Triggering Remote Execution"
echo "Iteration: $ITERATION"
echo "Instrument: $INSTRUMENT"
echo "========================================="

# Step 1: Verify local configs exist
echo ""
echo "[1/5] Verifying local configs..."

if [ ! -f "configs/iteration_${ITERATION}_train.yaml" ]; then
    echo "Error: Training config not found: configs/iteration_${ITERATION}_train.yaml"
    exit 1
fi

if [ ! -f "configs/iteration_${ITERATION}_backtest.yaml" ]; then
    echo "Error: Backtest config not found: configs/iteration_${ITERATION}_backtest.yaml"
    exit 1
fi

echo "✓ Local configs found"

# Step 2: Upload configs to remote server
echo ""
echo "[2/5] Uploading configs to remote server..."

scp -P ${REMOTE_PORT} \
  configs/iteration_${ITERATION}_train.yaml \
  ${REMOTE_HOST}:${REMOTE_PATH}/configs/

scp -P ${REMOTE_PORT} \
  configs/iteration_${ITERATION}_backtest.yaml \
  ${REMOTE_HOST}:${REMOTE_PATH}/configs/

echo "✓ Configs uploaded"

# Step 3: Upload remote_train_backtest.sh if not already present
echo ""
echo "[3/5] Ensuring remote script is available..."

scp -P ${REMOTE_PORT} \
  scripts/remote_train_backtest.sh \
  ${REMOTE_HOST}:${REMOTE_PATH}/scripts/

ssh -p ${REMOTE_PORT} ${REMOTE_HOST} \
  "chmod +x ${REMOTE_PATH}/scripts/remote_train_backtest.sh"

echo "✓ Remote script ready"

# Step 4: Execute remote training/backtesting
echo ""
echo "[4/5] Executing remote training and backtesting..."
echo "This may take ~45-50 minutes (training + backtest)..."
echo ""

ssh -p ${REMOTE_PORT} ${REMOTE_HOST} \
  "bash ${REMOTE_PATH}/scripts/remote_train_backtest.sh ${ITERATION} ${INSTRUMENT}"

REMOTE_EXIT_CODE=$?

if [ $REMOTE_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Error: Remote execution failed with exit code $REMOTE_EXIT_CODE"
    echo "Check remote logs for details"
    exit $REMOTE_EXIT_CODE
fi

echo ""
echo "✓ Remote execution complete"

# Step 5: Download results
echo ""
echo "[5/5] Downloading results..."

# Create local results directory
mkdir -p results/iteration_${ITERATION}

# Download backtest results
echo "  Downloading backtest results..."
scp -P ${REMOTE_PORT} \
  ${REMOTE_HOST}:${REMOTE_PATH}/backtest_results/results.json \
  results/iteration_${ITERATION}/results.json

# Download training results
echo "  Downloading training results..."
scp -P ${REMOTE_PORT} \
  ${REMOTE_HOST}:${REMOTE_PATH}/models/iteration_${ITERATION}/training_results.json \
  results/iteration_${ITERATION}/training_results.json

# Optional: Download model checkpoint (can be large ~50MB)
# Uncomment if you want to download the model
# echo "  Downloading model checkpoint..."
# scp -P ${REMOTE_PORT} \
#   ${REMOTE_HOST}:${REMOTE_PATH}/models/iteration_${ITERATION}/model.pth \
#   results/iteration_${ITERATION}/model.pth

echo "✓ Results downloaded"

# Step 6: Verify and display results
echo ""
echo "========================================="
echo "Results Summary"
echo "========================================="

# Verify JSON files are valid
python3 -c "import json; json.load(open('results/iteration_${ITERATION}/results.json'))" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Backtest results.json may be corrupted"
fi

python3 -c "import json; json.load(open('results/iteration_${ITERATION}/training_results.json'))" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Training results.json may be corrupted"
fi

# Display key metrics
echo ""
echo "Training Metrics:"
python3 -c "
import json
try:
    with open('results/iteration_${ITERATION}/training_results.json', 'r') as f:
        results = json.load(f)
    print(f\"  Initial Loss: {results['summary']['initial_loss']:.2f}\")
    print(f\"  Final Loss: {results['summary']['final_loss']:.2f}\")
    print(f\"  Improvement: {results['summary']['improvement_pct']:.1f}%\")
except Exception as e:
    print(f\"  Error: {e}\")
"

echo ""
echo "Backtest Metrics:"
python3 -c "
import json
try:
    with open('results/iteration_${ITERATION}/results.json', 'r') as f:
        results = json.load(f)
    deep = results['summary']['deep_hedge']
    bs = results['summary']['bs_baseline']
    print(f\"  Deep Hedge:\")
    print(f\"    Sharpe Ratio: {deep['sharpe_ratio']:.3f}\")
    print(f\"    CVaR 95%: {deep.get('cvar_95', 'N/A')}\")
    print(f\"    Mean PnL: {deep['mean']:.2f}\")
    print(f\"    Std PnL: {deep['std']:.2f}\")
    print(f\"  BS Baseline:\")
    print(f\"    Sharpe Ratio: {bs['sharpe_ratio']:.3f}\")
    print(f\"    CVaR 95%: {bs.get('cvar_95', 'N/A')}\")
except Exception as e:
    print(f\"  Error: {e}\")
"

echo ""
echo "========================================="
echo "Iteration $ITERATION Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - results/iteration_${ITERATION}/results.json"
echo "  - results/iteration_${ITERATION}/training_results.json"
echo ""
echo "Next step: Run Analysis Agent to generate optimization plan for iteration $((ITERATION + 1))"
echo ""
