# doom-agent

Reinforcement learning experiments for VizDoom (PPO, A2C, A3C, DQN variants, DDPG, and DreamerV3).

## Project layout

- `src/doom_agent/algorithms/`: Hierarchical organization of RL algorithms.
  - `ppo/`: PPO versions (base, v2, v3, v4, v5).
  - `dreamer/`: Model-based RL (DreamerV3).
  - `a2c/`: A2C variants (base, v1, v2, defend_center).
  - `dqn/`: DQN and DDDQN variants.
  - `a3c/`, `ddpg/`: Other implementations.
- `src/doom_agent/common/`: shared neural network helpers and monitoring utilities for Stable-Baselines agents.
- `src/doom_agent/wrappers/`: VizDoom environment wrappers (Gym-style and custom frame-stack wrappers).
- `src/doom_agent/scenarios/`: VizDoom scenario configs and WAD assets used across experiments.

## Highlights: DreamerV3

The latest and most advanced agent in this repository is **DreamerV3**, located in `src/doom_agent/algorithms/dreamer/v3/`.

- **Key Features**: Model-based imagination, Symmetry Augmentation (Mirror Learning), RGB Training, and detailed gameplay analytics.
- **Usage**:

  ```bash
  python src/doom_agent/algorithms/dreamer/v3/train.py --scenario deathmatch
  ```

## Highlights: PPO v5

A robust model-free baseline with sequential curriculum training.

- **Location**: `src/doom_agent/algorithms/ppo/v5/`.
- **Usage**:

  ```bash
  python src/doom_agent/algorithms/ppo/v5/train.py --scenario deadly_corridor
  ```

## 📊 SOTA Benchmarking: Arnold

This repository includes an integration of the **Arnold agent** (2017 ViZDoom Champion) as a State-of-the-Art benchmark.

- **Location**: `external/arnold/` and `scripts/arnold_adapter.py`.
- **Validation**: Compare Arnold's behavior against your reward shaping incentives using the "BOSS MODE" SPECTATOR.
- **Usage**:

  ```bash
  python scripts/test_reward_shaping.py --arnold
  ```

## 🧠 SOTA Benchmarking: Intel DFP

In addition to Arnold, we have integrated the **Intel Direct Future Prediction (DFP)** agent, winner of the 2016 ViZDoom Track 1.

- **Location**: `external/dfp/` and `scripts/dfp_adapter.py`.
- **Compatibility**: Legacy TensorFlow 1.x code automatically patched for TF 2.x execution using `compat.v1`.
- **Usage**:

  ```bash
  # First time setup
  bash scripts/setup_dfp.sh
  # Run benchmark
  python scripts/test_reward_shaping.py --dfp
  ```

## 🧪 Reward Shaping Validator

A real-time tool to validate incentives (Hunger, Movement, Frags) before long training runs.

- **Manual Mode**: Test the feel of your rewards by playing as the agent.
- **AI Mode**: Evaluate a specific DreamerV3 checkpoint.
- **Arnold Mode**: Benchmark against the legendary champion.
- **Usage**:

  ```bash
  # Manual control (W/S/A/D/Q/E/Space)
  python scripts/test_reward_shaping.py --manual

  # AI Spectator (DreamerV3)
  python scripts/test_reward_shaping.py --agent_run <RUN_ID>
  ```

## 📚 Scientific Foundation

This project is built upon several seminal works in the field of Reinforcement Learning. For a detailed explanation of the theory behind DreamerV3 and the historical SOTA agents (Arnold, DFP), please refer to the **[THEORY.md](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/THEORY.md)** documentation.

### Core References

- **DreamerV3**: Hafner et al. (2023)
- **Arnold**: Lample & Chaplot (2017)
- **Direct Future Prediction**: Dosovitskiy & Koltun (2016)

## Running code

1) Install dependencies: `pip install -r requirements_tf_gpu.yaml` (legacy) or ensure `torch`, `vizdoom`, `opencv-python` are installed.
2) Expose the source layout: `export PYTHONPATH="$(pwd)/src"`.
3) Run an experiment, e.g.:
   - `python src/doom_agent/algorithms/dreamer/v3/train.py --scenario deathmatch`
   - `python src/doom_agent/algorithms/ppo/v4/train_deadly_corridor.py` (Stable-Baselines3)

Scenario paths are resolved through `doom_agent.paths.scenario_path`, so experiments can be launched from the repository root without changing directories.
