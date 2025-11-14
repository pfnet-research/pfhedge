#!/bin/bash
# Download backtest results and trained models from remote GPU server
# Usage: ./download_results.sh <iteration_number> <ssh_host> <ssh_port> [--download-model]

set -e

ITERATION=$1
SSH_HOST=$2
SSH_PORT=$3
DOWNLOAD_MODEL=$4

if [ -z "$ITERATION" ] || [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "Usage: $0 <iteration_number> <ssh_host> <ssh_port> [--download-model]"
    echo "Example: $0 21 185.65.93.114 47612 --download-model"
    exit 1
fi

echo "=== Downloading Iteration $ITERATION Results ==="
echo "Remote: root@$SSH_HOST:$SSH_PORT"
echo ""

# Create local directories
LOCAL_RESULTS_DIR="results/iteration_$ITERATION"
LOCAL_MODELS_DIR="models"
mkdir -p "$LOCAL_RESULTS_DIR"
mkdir -p "$LOCAL_MODELS_DIR"

# Download backtest report (markdown)
echo "[1/5] Downloading backtest report..."
scp -P "$SSH_PORT" "root@$SSH_HOST:/workspace/pfhedge/backtest_results/iteration_$ITERATION/backtest_report.md" \
    "$LOCAL_RESULTS_DIR/" || echo "Warning: Report not found"

# Download backtest metrics (JSON)
echo "[2/5] Downloading metrics..."
scp -P "$SSH_PORT" "root@$SSH_HOST:/workspace/pfhedge/backtest_results/iteration_$ITERATION/backtest_metrics.json" \
    "$LOCAL_RESULTS_DIR/" || echo "Warning: Metrics not found"

# Download raw backtest data
echo "[3/5] Downloading raw data..."
scp -P "$SSH_PORT" "root@$SSH_HOST:/workspace/pfhedge/backtest_results/iteration_$ITERATION/raw_data.pkl" \
    "$LOCAL_RESULTS_DIR/" || echo "Warning: Raw data not found"

# Download all plots
echo "[4/5] Downloading plots..."
mkdir -p "$LOCAL_RESULTS_DIR/plots"
scp -P "$SSH_PORT" "root@$SSH_HOST:/workspace/pfhedge/backtest_results/iteration_$ITERATION/plots/*" \
    "$LOCAL_RESULTS_DIR/plots/" || echo "Warning: Plots not found"

# Download trained model if requested
if [ "$DOWNLOAD_MODEL" == "--download-model" ]; then
    echo "[5/5] Downloading trained model..."

    # Get model directory with git hash
    MODEL_DIR=$(ssh -p "$SSH_PORT" "root@$SSH_HOST" "ls -d /workspace/pfhedge/models/iteration_${ITERATION}_* 2>/dev/null | head -1")

    if [ -n "$MODEL_DIR" ]; then
        MODEL_NAME=$(basename "$MODEL_DIR")
        echo "Found model: $MODEL_NAME"

        mkdir -p "$LOCAL_MODELS_DIR/$MODEL_NAME"

        # Download model files
        scp -P "$SSH_PORT" "root@$SSH_HOST:$MODEL_DIR/model.pth" \
            "$LOCAL_MODELS_DIR/$MODEL_NAME/" || echo "Error: model.pth not found"

        scp -P "$SSH_PORT" "root@$SSH_HOST:$MODEL_DIR/config.yaml" \
            "$LOCAL_MODELS_DIR/$MODEL_NAME/" || echo "Warning: config.yaml not found"

        scp -P "$SSH_PORT" "root@$SSH_HOST:$MODEL_DIR/training_log.txt" \
            "$LOCAL_MODELS_DIR/$MODEL_NAME/" || echo "Warning: training_log.txt not found"

        echo "Model downloaded to: $LOCAL_MODELS_DIR/$MODEL_NAME"
    else
        echo "Error: No model directory found for iteration $ITERATION"
        exit 1
    fi
else
    echo "[5/5] Skipping model download (use --download-model to enable)"
fi

echo ""
echo "=== Download Complete ==="
echo "Results location: $LOCAL_RESULTS_DIR"
if [ "$DOWNLOAD_MODEL" == "--download-model" ]; then
    echo "Model location: $LOCAL_MODELS_DIR/iteration_${ITERATION}_*"
fi
echo ""

# Display metrics summary if available
if [ -f "$LOCAL_RESULTS_DIR/backtest_metrics.json" ]; then
    echo "=== Metrics Summary ==="
    python3 -c "
import json
import sys
try:
    with open('$LOCAL_RESULTS_DIR/backtest_metrics.json') as f:
        metrics = json.load(f)
    dh = metrics.get('deep_hedge', {})
    print(f\"Sharpe Ratio: {dh.get('sharpe_ratio', 'N/A')}\")
    print(f\"CVaR (95%): \${dh.get('cvar_95', 'N/A')}\")
    print(f\"Mean PnL: \${dh.get('mean_pnl', 'N/A')}\")
    print(f\"Win Rate: {dh.get('win_rate', 'N/A')}\")
except Exception as e:
    print(f'Could not parse metrics: {e}', file=sys.stderr)
"
fi
