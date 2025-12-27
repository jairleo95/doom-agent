#!/bin/bash
# High-Performance Runner for RTX 6000 Ada (48GB VRAM)
# Optimized for maximum throughput and large batch training

export PYTHONPATH=$(pwd)/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# BULLETPROOF STABILITY CONFIGURATION (48GB VRAM):
# - n-envs 16: Standard worker count for predictable overhead
# - batch-size 128: Guaranteed to fit comfortably in 48GB VRAM
# - batch-length 64: High temporal context maintained
# - train-every 128: Optimized train/collect ratio for high FPS
# - train-steps 1: Efficient training step
python src/doom_agent/algorithms/dreamer_v3/train.py \
  --scenario deathmatch_curriculum \
  --n-envs 16 \
  --device cuda \
  --batch-size 128 \
  --batch-length 64 \
  --train-every 128 \
  --train-steps 1 \
  --prefill-steps 20000 \
  --video-freq 100000
