#!/bin/bash
# Optimized runner for RTX 5090 (32GB VRAM) + ~15 vCPUs
# Use this on RunPod/Cloud instances after running setup_cloud.sh

export PYTHONPATH=$(pwd)/src:$(pwd)/src/doom_agent/algorithms/dreamer_v3/nm512_dreamer
python src/doom_agent/algorithms/dreamer_v3/train.py \
  --scenario deathmatch \
  --n-envs 12 \
  --device cuda \
  --batch-size 64 \
  --train-every 5 \
  --train-steps 5
