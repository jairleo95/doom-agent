#!/bin/bash

# DreamerV3 20-Minute Verification Run (Improved)
# Aimed at testing BC + RL pipeline with meaningful steps in a short window.
# Runs two variants to verify the statistical analysis script.

SCENARIO="deathmatch"
STEPS=10000
BC_STEPS=5000
N_ENVS=12
COMMON_ARGS="scenario=$SCENARIO agent.n_envs=$N_ENVS device=cuda wandb.enabled=false"

echo "=========================================================="
echo "STARTING VERIFICATION SUITE (~20 MINS TOTAL)"
echo "Target: $STEPS steps per run, $BC_STEPS BC steps, $N_ENVS envs"
echo "=========================================================="

# 1. Run Baseline (with BC)
echo "--- Running Variant 1: Baseline (with BC) ---"
PYTHONPATH=./src python src/doom_agent/algorithms/dreamer/v3/train.py \
    $COMMON_ARGS \
    agent.pretrain_bc=true \
    agent.pretrain_steps=$BC_STEPS \
    agent.pretrain_bc_path=data/expert_replays \
    hydra.run.dir="results/verification/baseline" \
    scenario.curriculum.stages.0.timesteps=$STEPS

# 2. Run No-BC Variant
echo "--- Running Variant 2: No-BC ---"
PYTHONPATH=./src python src/doom_agent/algorithms/dreamer/v3/train.py \
    $COMMON_ARGS \
    agent.pretrain_bc=false \
    hydra.run.dir="results/verification/no_bc" \
    scenario.curriculum.stages.0.timesteps=$STEPS

echo "Verification suite complete! Analyzing results..."
PYTHONPATH=./src python scripts/compare_ablations.py --dir results/verification --baseline baseline
