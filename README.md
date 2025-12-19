# doom-agent

Reinforcement learning experiments for VizDoom (PPO, A2C, A3C, DQN variants, DDPG, and playground examples).

## Project layout
- `src/doom_agent/algorithms/`: individual experiment folders (ppo, ppo_v2, ppo_v3, a2c, a2c_v1, a2c_v2, a2c_defend_the_center, a3c, dqn, dqn_tf2, dddqn_tf2, dddqn_tf2_v2, ddpg_tf2).
- `src/doom_agent/common/`: shared neural network helpers and monitoring utilities for Stable-Baselines agents.
- `src/doom_agent/utils/`: replay buffers, plotting utilities, parameter helpers, and GPU memory configuration helpers.
- `src/doom_agent/wrappers/`: VizDoom environment wrappers (Gym-style and custom frame-stack wrappers).
- `src/doom_agent/examples/`: example scripts (e.g., Stable-Baselines PPO on basic_gym).
- `src/doom_agent/scenarios/`: VizDoom scenario configs and WAD assets used across experiments.
- `src/doom_agent/algorithms/ppo_v4/`: PPO training using Stable-Baselines3 (no custom PPO code).

## Running code
1) Install dependencies (adjust to your CUDA/CPU setup): `pip install -r requirements_tf_gpu.yaml`.
2) Expose the source layout: `export PYTHONPATH="$(pwd)/src"`.
3) Run an experiment, e.g.:
   - `python -m doom_agent.algorithms.ppo_v3.ppo_vizdoom_deadly_corridor`
   - Evaluar PPO v3 entrenado: `python -m doom_agent.algorithms.ppo_v3.eval_deadly_corridor --model checkpoints/deadly_corridor/best.pth`
   - `python -m doom_agent.algorithms.ppo.ppo_vizdoom_basic`
   - `python -m doom_agent.algorithms.a2c.main`
   - `python -m doom_agent.algorithms.ppo_v4.train_deadly_corridor` (uses Stable-Baselines3 PPO)
   - `python -m doom_agent.algorithms.ppo_v4.train_defend_the_center` (SB3 PPO con curriculum simple)
   - Evaluar PPO v4 defend_the_center: `python -m doom_agent.algorithms.ppo_v4.eval_defend_the_center --model checkpoints/defend_the_center/ppo_v4_defend_final.zip`
   - Evaluar PPO v4 entrenado: `python -m doom_agent.algorithms.ppo_v4.eval_deadly_corridor --model checkpoints/deadly_corridor/ppo_v4_best.zip`

Scenario paths are now resolved through `doom_agent.paths.scenario_path`, so experiments can be launched from the repository root without changing directories.
