#!/bin/bash
# High-Performance Runner for RTX 6000 Ada (48GB VRAM)
# Optimized for maximum throughput and large batch training

export PYTHONPATH=$(pwd)/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ROCK-SOLID STABILITY CONFIGURATION (48GB VRAM):
# - n-envs 16: Reduced worker overhead
# - batch-size 256: Safe memory footprint for 48GB VRAM
# - batch-length 64: High temporal context maintained
# - train-every 1024: Efficient collection buffer
# - train-steps 1: Efficient training step
python src/doom_agent/algorithms/dreamer_v3/train.py \
  --scenario deathmatch_curriculum \
  --n-envs 16 \
  --device cuda \
  --batch-size 256 \
  --batch-length 64 \
  --train-every 1024 \
  --train-steps 1 \
  --prefill-steps 20000 \
  --video-freq 100000
