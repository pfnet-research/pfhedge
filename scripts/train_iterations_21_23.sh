#!/bin/bash
# Sequential training and backtesting for iterations 21 and 23
# Run this on remote GPU server with: bash scripts/train_iterations_21_23.sh

set -e  # Exit on error

PYTHON="/venv/main/bin/python"
REPO_DIR="/workspace/pfhedge"
OPTION_FILE="crypto/data/options_2025-09-28_monthly_atm_110k.json"
INSTRUMENT="BTC-31OCT25-110000-C"

cd $REPO_DIR

echo "=========================================="
echo "Starting Sequential Training Pipeline"
echo "Time: $(date)"
echo "=========================================="

# Function to find model directory with git hash
find_model_dir() {
    local iteration=$1
    local model_dir=$(ls -td models/iteration_${iteration}* 2>/dev/null | head -1)
    if [ -z "$model_dir" ]; then
        echo "ERROR: No model directory found for iteration $iteration" >&2
        exit 1
    fi
    echo "$model_dir"
}

# Function to update backtest config with absolute model path
update_backtest_config() {
    local iteration=$1
    local model_dir=$2
    local config="configs/iteration_${iteration}_backtest.yaml"

    echo "Updating $config with model path: ${REPO_DIR}/${model_dir}/model.pth"
    sed -i "s|model_path: .*|model_path: \"${REPO_DIR}/${model_dir}/model.pth\"|" "$config"

    # Also ensure absolute paths for data
    sed -i "s|data_dir: \"crypto/data/historical\"|data_dir: \"${REPO_DIR}/crypto/data/historical\"|" "$config"
    sed -i "s|data_dir: crypto/data/historical|data_dir: \"${REPO_DIR}/crypto/data/historical\"|" "$config"
}

echo ""
echo "=========================================="
echo "ITERATION 21: Training"
echo "=========================================="
echo "Start time: $(date)"

$PYTHON crypto/scripts/train_for_option.py \
    --config configs/iteration_21_train.yaml \
    --option-file $OPTION_FILE \
    --instrument $INSTRUMENT

echo "Training completed: $(date)"
echo "Finding model directory..."
MODEL_21=$(find_model_dir 21)
echo "Found model: $MODEL_21"

echo ""
echo "=========================================="
echo "ITERATION 21: Backtesting"
echo "=========================================="
echo "Start time: $(date)"

update_backtest_config 21 "$MODEL_21"

$PYTHON -m crypto.backtest --config configs/iteration_21_backtest.yaml

echo "Backtest completed: $(date)"
echo "Results saved to: ${REPO_DIR}/backtest_results/iteration_21/"

echo ""
echo "=========================================="
echo "ITERATION 23: Training"
echo "=========================================="
echo "Start time: $(date)"

$PYTHON crypto/scripts/train_for_option.py \
    --config configs/iteration_23_train.yaml \
    --option-file $OPTION_FILE \
    --instrument $INSTRUMENT

echo "Training completed: $(date)"
echo "Finding model directory..."
MODEL_23=$(find_model_dir 23)
echo "Found model: $MODEL_23"

echo ""
echo "=========================================="
echo "ITERATION 23: Backtesting"
echo "=========================================="
echo "Start time: $(date)"

update_backtest_config 23 "$MODEL_23"

$PYTHON -m crypto.backtest --config configs/iteration_23_backtest.yaml

echo "Backtest completed: $(date)"
echo "Results saved to: ${REPO_DIR}/backtest_results/iteration_23/"

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Iteration 21 model: $MODEL_21"
echo "  - Iteration 21 results: backtest_results/iteration_21/"
echo "  - Iteration 23 model: $MODEL_23"
echo "  - Iteration 23 results: backtest_results/iteration_23/"
echo ""
echo "Next steps:"
echo "  1. Download results: python scripts/download_results.py 21 185.65.93.114 47612"
echo "  2. Download results: python scripts/download_results.py 23 185.65.93.114 47612"
