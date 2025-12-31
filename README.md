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

## Running code
1) Install dependencies: `pip install -r requirements_tf_gpu.yaml` (legacy) or ensure `torch`, `vizdoom`, `opencv-python` are installed.
2) Expose the source layout: `export PYTHONPATH="$(pwd)/src"`.
3) Run an experiment, e.g.:
   - `python src/doom_agent/algorithms/dreamer/v3/train.py --scenario deathmatch`
   - `python src/doom_agent/algorithms/ppo/v4/train_deadly_corridor.py` (Stable-Baselines3)

Scenario paths are resolved through `doom_agent.paths.scenario_path`, so experiments can be launched from the repository root without changing directories.
