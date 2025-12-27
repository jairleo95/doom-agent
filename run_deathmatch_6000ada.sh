#!/bin/bash
# High-Performance Runner for RTX 6000 Ada (48GB VRAM)
# Optimized for maximum throughput and large batch training

export PYTHONPATH=$(pwd)/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HIGH-STABILITY CONFIGURATION (48GB VRAM):
# - n-envs 24: High collections frequency
# - batch-size 384: Stable under 48GB VRAM for temporal sequences
# - batch-length 64: Increased temporal context
# - train-every 1024: Massive collection buffer for high FPS
# - train-steps 1: Efficient training step
python src/doom_agent/algorithms/dreamer_v3/train.py \
  --scenario deathmatch_curriculum \
  --n-envs 24 \
  --device cuda \
  --batch-size 384 \
  --batch-length 64 \
  --train-every 1024 \
  --train-steps 1 \
  --prefill-steps 20000 \
  --video-freq 100000
