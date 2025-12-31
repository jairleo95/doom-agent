#!/bin/bash
# PEAK PERFORMANCE Runner for RTX 6000 Ada / 96GB PRO
# Maximizes throughput using torch.compile, large batches, and 32 parallel environments.

export PYTHONPATH=$(pwd)/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Launching Dreamer V3 PEAK PERFORMANCE mode..."
echo "Config: 32 Environments | Batch 512 | torch.compile=Enabled"

python src/doom_agent/algorithms/dreamer/v3/train.py \
  scenario=deathmatch_curriculum \
  hardware=rtx6000_peak \
  wandb.enabled=true \
  wandb.group=dreamer-v3-peak
