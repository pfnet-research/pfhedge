#!/bin/bash

# Parse command line arguments
FORCE=false
if [ "$1" = "--force" ] || [ "$1" = "-f" ]; then
    FORCE=true
    echo "⚠️  Force mode enabled: will overwrite existing results"
fi

echo "Starting batch backtesting..."
echo "============================================================"

total_models=$(find results/hparam_tuning -name 'model.pth' 2>/dev/null | wc -l)
echo "Total trained models found: $total_models"

if [ "$total_models" -eq 0 ]; then
    echo ""
    echo "❌ No trained models found!"
    echo ""
    echo "Please run training first:"
    echo "  bash run_all_training.sh"
    echo ""
    exit 1
fi

echo

completed=0
skipped=0
failed=0

for model_path in results/hparam_tuning/*/train_*/model.pth; do
    if [ ! -f "$model_path" ]; then
        continue
    fi

    train_dir=$(dirname "$model_path")
    run_dir=$(dirname "$train_dir")
    run_name=$(basename "$run_dir")

    # Extract git hash from train directory name (train_<hash>)
    git_hash=$(basename "$train_dir" | sed 's/train_//')

    # Backtest output directory
    backtest_dir="$run_dir/backtest_$git_hash"

    # Check if backtest already completed (skip if not forcing)
    if [ "$FORCE" = false ] && [ -f "$backtest_dir/results.json" ]; then
        echo "⏭  Skipping (already backtested): $run_name"
        ((skipped++))
        continue
    fi

    # Check if training results exist to get config
    if [ ! -f "$train_dir/training_results.json" ]; then
        echo "⚠️  No training_results.json: $run_name"
        ((failed++))
        continue
    fi

    # Create backtest config from training results
    backtest_config="$train_dir/backtest_config.yaml"

    if [ ! -f "$backtest_config" ] || [ "$FORCE" = true ]; then
        echo "Creating backtest config: $run_name"
        python crypto/scripts/create_backtest_config.py \
            --train-results "$train_dir/training_results.json" \
            --option-metadata "$train_dir/option_metadata.json" \
            --model-path "$model_path" \
            --template backtest.yaml \
            --output "$backtest_config" \
            --output-dir "$backtest_dir"

        if [ $? -ne 0 ]; then
            echo "✗ Failed to create config: $run_name"
            ((failed++))
            continue
        fi
    fi

    echo "============================================================"
    echo "Running backtest: $run_name"
    echo "Model: $model_path"
    echo "Config: $backtest_config"
    echo "Started: $(date)"
    echo "============================================================"

    # Run backtest (all config from file)
    python -m crypto.backtest.run --config "$backtest_config"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed backtest: $run_name"
        ((completed++))
    else
        echo "✗ Failed backtest: $run_name (exit code: $exit_code)"
        ((failed++))
    fi
    echo
done

echo "============================================================"
echo "All backtests completed!"
echo "Summary:"
echo "  Completed: $completed"
echo "  Skipped: $skipped (already backtested)"
echo "  Failed: $failed"
echo "  Total: $((completed + skipped + failed))"
echo "============================================================"
