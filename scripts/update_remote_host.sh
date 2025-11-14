#!/bin/bash
# One-shot script to update remote host IP and port across all scripts and docs
# Usage: bash scripts/update_remote_host.sh <new_host> <new_port>

set -e

NEW_HOST=$1
NEW_PORT=$2

if [ -z "$NEW_HOST" ] || [ -z "$NEW_PORT" ]; then
    echo "Usage: bash scripts/update_remote_host.sh <new_host> <new_port>"
    echo "Example: bash scripts/update_remote_host.sh 185.65.93.114 47612"
    exit 1
fi

echo "========================================="
echo "Updating Remote Host Configuration"
echo "========================================="
echo "New Host: $NEW_HOST"
echo "New Port: $NEW_PORT"
echo ""

# Files to update
FILES=(
    "scripts/trigger_remote.sh"
    "scripts/remote_train_backtest.sh"
    "scripts/agents/test_agent_prompt.md"
    "scripts/ORCHESTRATOR_PROMPT.md"
)

# Old patterns to replace (will be detected automatically)
OLD_HOSTS=("185.65.93.212" "185.65.93.114" "174.78.228.101" "40.84.35.44")
OLD_PORTS=("43763" "40532" "7523" "47980" "47612")

echo "Updating files..."
echo ""

for FILE in "${FILES[@]}"; do
    if [ ! -f "$FILE" ]; then
        echo "Warning: $FILE not found, skipping..."
        continue
    fi

    echo "Processing: $FILE"

    # Backup original file
    cp "$FILE" "$FILE.bak"

    # Replace all old hosts with new host
    for OLD_HOST in "${OLD_HOSTS[@]}"; do
        if grep -q "$OLD_HOST" "$FILE" 2>/dev/null; then
            sed -i '' "s/$OLD_HOST/$NEW_HOST/g" "$FILE"
            echo "  ✓ Replaced $OLD_HOST with $NEW_HOST"
        fi
    done

    # Replace all old ports with new port
    for OLD_PORT in "${OLD_PORTS[@]}"; do
        if grep -q "$OLD_PORT" "$FILE" 2>/dev/null; then
            sed -i '' "s/$OLD_PORT/$NEW_PORT/g" "$FILE"
            echo "  ✓ Replaced port $OLD_PORT with $NEW_PORT"
        fi
    done

    # Remove backup if changes were successful
    rm "$FILE.bak"
    echo ""
done

echo "========================================="
echo "Update Complete!"
echo "========================================="
echo ""
echo "Changed files:"
git diff --name-only
echo ""
echo "Review changes with: git diff"
echo "Commit changes with: git add -A && git commit -m 'Update remote host to $NEW_HOST:$NEW_PORT'"
echo ""
