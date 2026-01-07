#!/bin/bash

# Usage: ./visualize_from_wandb.sh RUN_ID
# Example: ./visualize_from_wandb.sh 20260101-054831_dreamer

if [ -z "$1" ]; then
    echo "Usage: ./visualize_from_wandb.sh RUN_ID"
    exit 1
fi

RUN_ID=$1

# 1. Download model
echo "--- Downloading Model from W&B ---"
python scripts/download_model.py --run_id $RUN_ID --project doom-agent

# 2. Get the path (the script prints it, but we can find it)
MODEL_PATH=$(find downloads/$RUN_ID -name "*.pt" | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model file not found after download."
    exit 1
fi

# 3. Visualize
echo "--- Starting Visualization ---"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/doom_agent/algorithms/dreamer/v3/visualize.py \
    --path $MODEL_PATH \
    --scenario deathmatch \
    --fps 35
