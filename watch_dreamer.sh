#!/bin/bash
# Convenient runner for DreamerV3 visualization

export PYTHONPATH=$(pwd)/src

# Usage: ./watch_dreamer.sh [path_to_run_or_checkpoint]
# If no path is provided, it tries to find the most recent deathmatch_curriculum run.

TARGET_PATH=$1

if [ -z "$TARGET_PATH" ]; then
    # Find latest run directory in checkpoints
    LATEST_RUN=$(ls -td src/doom_agent/algorithms/dreamer/v3/checkpoints/deathmatch_curriculum/*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "Error: No recent deathmatch_curriculum runs found in checkpoints."
        echo "Usage: ./watch_dreamer.sh src/doom_agent/algorithms/dreamer/v3/checkpoints/deathmatch_curriculum/RUN_ID"
        exit 1
    fi
    TARGET_PATH=$LATEST_RUN
    echo "No path provided. Using latest run: $TARGET_PATH"
fi

python src/doom_agent/algorithms/dreamer/v3/visualize.py \
    --path "$TARGET_PATH" \
    --scenario deathmatch_curriculum \
    --episodes 5 \
    --fps 35
