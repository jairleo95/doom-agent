#!/bin/bash
# Doom Agent - H200 High Performance Launch Script
# Optimized for Cloud Environments (RunPod/Lambda/Lambda Stack)

echo "--- Initializing Dreamer V3 on NVIDIA H200 (141GB) ---"

# 1. Setup Environment
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # Determinism for debugging
export TORCH_COMPILE_BACKEND="inductor" # Fastest for Hopper

# 2. Check GPU Specs
nvidia-smi

# 3. Launch Training
# Hardware: H200 overrides (Large Batch, Large Model)
# Environment: RGB Training by default for SOTA quality
python src/doom_agent/algorithms/dreamer/v3/train.py \
    hardware=h200 \
    env.rgb=true \
    wandb.name="h200_sota_deathmatch_$(date +%Y%m%d_%H%M%S)" \
    wandb.mode=online \
    agent.compile=true

echo "--- Run Finished ---"
