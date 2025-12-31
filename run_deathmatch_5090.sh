#!/bin/bash
# Optimized runner for RTX 5090 (32GB VRAM) + ~15 vCPUs
# Use this on RunPod/Cloud instances after running setup_cloud.sh

export PYTHONPATH=$(pwd)/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/doom_agent/algorithms/dreamer/v3/train.py \
  --scenario deathmatch \
  --n-envs 16 \
  --device cuda \
  --batch-size 128 \
  --train-every 128 \
  --train-steps 1 \
  --prefill-steps 15000
