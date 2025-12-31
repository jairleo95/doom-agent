#!/bin/bash
# Local high-performance script for RTX 3060 (12GB VRAM)
# Optimized to squeeze maximum throughput without OOM

export PYTHONPATH=$(pwd)/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Balanced settings for 12GB VRAM to avoid OOM during training:
python src/doom_agent/algorithms/dreamer/v3/train.py \
  --scenario deathmatch_curriculum \
  --n-envs 8 \
  --device cuda \
  --batch-size 32 \
  --batch-length 32 \
  --train-every 10 \
  --train-steps 1 \
  --prefill-steps 5000 \
  --video-freq 50000
