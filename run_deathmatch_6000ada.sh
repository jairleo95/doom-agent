#!/bin/bash
# High-Performance Runner for RTX 6000 Ada (48GB VRAM)
# Optimized for maximum throughput and large batch training

export PYTHONPATH=$(pwd)/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MAXIMUM CAPACITY CONFIGURATION:
# - n-envs 32: Fully utilizes 16+ vCPUs
# - batch-size 512: Leverages 48GB VRAM for high-quality updates
# - batch-length 64: Increased temporal context
# - train-every 1024: Massive collection buffer for high FPS
# - train-steps 1: Efficient training step
python src/doom_agent/algorithms/dreamer_v3/train.py \
  --scenario deathmatch_curriculum \
  --n-envs 32 \
  --device cuda \
  --batch-size 512 \
  --batch-length 64 \
  --train-every 1024 \
  --train-steps 1 \
  --prefill-steps 20000 \
  --video-freq 100000
