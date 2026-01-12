#!/bin/bash

# Mini Ablation Test
# Verifies the full pipeline with minimal steps.

SCENARIO="deathmatch"
STEPS=1000
BC_STEPS=100
COMMON_ARGS="scenario=$SCENARIO agent.pretrain_bc_path=data/expert_replays device=cuda agent.n_envs=1 wandb.enabled=false"

# Run Baseline
PYTHONPATH=./src python src/doom_agent/algorithms/dreamer/v3/train.py \
    $COMMON_ARGS \
    wandb.name="ablation_test_baseline" \
    hydra.run.dir="results/ablations/test_baseline" \
    agent.pretrain_bc=true agent.pretrain_steps=$BC_STEPS \
    scenario.curriculum.stages.0.timesteps=$STEPS

# Run No BC
PYTHONPATH=./src python src/doom_agent/algorithms/dreamer/v3/train.py \
    $COMMON_ARGS \
    wandb.name="ablation_test_no_bc" \
    hydra.run.dir="results/ablations/test_no_bc" \
    agent.pretrain_bc=false \
    scenario.curriculum.stages.0.timesteps=$STEPS

# Run Comparison (redirecting results to a specific test folder)
python scripts/compare_ablations.py --dir results/ablations
