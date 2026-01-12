#!/bin/bash

# DreamerV3 Ablation Suite Launcher
# This script runs sequential experiments to isolate the impact of different components.

SCENARIO="deathmatch"
STEPS_PER_STAGE=100000
BC_STEPS=50000
COMMON_ARGS="scenario=$SCENARIO agent.pretrain_bc_path=data/expert_replays device=cuda wandb.enabled=true"

# Helper function to run an experiment
run_exp() {
    NAME=$1
    OVERRIDES=$2
    echo "=========================================================="
    echo "RUNNING ABLATION: $NAME"
    echo "=========================================================="
    
    PYTHONPATH=./src python src/doom_agent/algorithms/dreamer/v3/train.py \
        $COMMON_ARGS \
        wandb.name="ablation_$NAME" \
        hydra.run.dir="results/ablations/$NAME" \
        $OVERRIDES
}

# 1. Baseline (Full Model)
run_exp "baseline" "agent.symmetry=true agent.reward_shaping=true agent.pretrain_bc=true agent.pretrain_steps=$BC_STEPS"

# 2. No Symmetry (Mirror Learning)
run_exp "no_symmetry" "agent.symmetry=false agent.reward_shaping=true agent.pretrain_bc=true agent.pretrain_steps=$BC_STEPS"

# 3. No Reward Shaping (Frags Only)
run_exp "no_rewards" "agent.symmetry=true agent.reward_shaping=false agent.pretrain_bc=true agent.pretrain_steps=$BC_STEPS"

# 4. Small RSSM Capacity (128 units)
run_exp "small_rssm" "agent.symmetry=true agent.reward_shaping=true agent.dyn_deter=128 agent.dyn_hidden=128 agent.units=128 agent.pretrain_bc=true agent.pretrain_steps=$BC_STEPS"

# 5. No Imitation Learning (Zero-Shot RL)
run_exp "no_bc" "agent.symmetry=true agent.reward_shaping=true agent.pretrain_bc=false agent.pretrain_steps=0"

echo "Ablation Suite Complete! Check results/ablations/ and W&B."
