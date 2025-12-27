#!/bin/bash
# Optimized runner for RTX 5090 (32GB VRAM) + ~15 vCPUs
# Use this on RunPod/Cloud instances after running setup_cloud.sh

export PYTHONPATH=$(pwd)/src
python src/doom_agent/algorithms/dreamer_v3/train.py \
  --scenario deathmatch \
  --n-envs 64 \
  --device cuda \
  --batch-size 512 \
  --train-every 10 \
  --train-steps 10 \
  --prefill-steps 20000
