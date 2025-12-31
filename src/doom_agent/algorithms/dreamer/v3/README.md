# Dreamer V3 for Doom

Model-based reinforcement learning implementation using Dreamer V3 for Doom scenarios, using Hydra for configuration and W&B for experiment tracking.

## Architecture

This implementation follows a SOLID modular structure to ensure maintainability and scalability:

### Core Components

- **[train.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/train.py)**: Slim entry point that initializes Hydra and orchestrates the high-level experiment lifecycle.
- **[trainer.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/trainer.py)**: Encapsulates the `DreamerV3Trainer` class, managing the environment lifecycle, parallel execution, and the main curriculum training loop.
- **[experiment.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/experiment.py)**: Contains `ExperimentManager`, responsible for filesystem organization, configuration persistence, and W&B integration.
- **[agent.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/agent.py)**: Adapter for the DreamerV3 RSSM model.
- **[utils.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/utils.py)**: Shared utility functions for action flipping and logging.

### Files & Modules

- **[callbacks/](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/callbacks/)** - Training callbacks package
  - `video.py` - Records high-res episode videos
  - `imagination.py` - Logs world model predictions
  - `metrics_logger.py` - Logs training and gameplay metrics
  - `checkpoint.py` - Manages model saving
  - `evaluation.py` - Handles periodic evaluation
- **[curriculum.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/curriculum.py)** - Multi-stage training stage definitions.
- **[doom_envs.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/doom_envs.py)** - RGB environment wrappers with high-fidelity metric collection.
- **[replay_buffer.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/replay_buffer.py)** - Experience replay with symmetry augmentation.

## Usage

This project uses **Hydra** for configuration management. Use the key=value syntax to override any parameters.

### Basic Training

Train on Deathmatch with default curriculum:

```bash
# From project root
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/doom_agent/algorithms/dreamer/v3/train.py scenario=deathmatch hardware=rtx3060
```

### Advanced Overrides

```bash
python src/doom_agent/algorithms/dreamer/v3/train.py \
  scenario=deathmatch \
  hardware=rtx4090 \
  agent.batch_size=128 \
  agent.n_envs=16 \
  wandb.enabled=true \
  video_freq=50000
```

### Resuming Training

```bash
python src/doom_agent/algorithms/dreamer/v3/train.py \
  scenario=deathmatch \
  resume=/path/to/checkpoint.pt \
  start_stage=1
```

## Key Features

- **SOLID Refactoring**: Decoupled orchestration for clean code and easy debugging.
- **Hydra Integration**: Hierarchical configuration for hardware profiles and scenarios.
- **W&B Artifacts**: Automatic upload of "Best Model" and "Stage Final" checkpoints to the cloud.
- **Model-Based (DreamerV3)**: High sample efficiency using latent imagination.
- **Symmetry Augmentation**: Mirror learning (Horizontal Flip) to double data efficiency.
- **RGB Training**: Full-color high-fidelity vision support.

## Output Structure

Experiments are organized by scenario and timestamp:

```
src/doom_agent/algorithms/dreamer/v3/
├── runs/               # Experiment logs and metadata
│   └── scenario_name/
│       └── YYYYMMDD-HHMMSS_dreamer/
│           ├── config.json
│           └── [stage_name].json
├── checkpoints/        # Model checkpoints and videos
│   └── scenario_name/
│       └── YYYYMMDD-HHMMSS_dreamer/
│           ├── [stage_name]/
│           └── videos/
```

## Testing

Run the full test suite using `pytest`:

```bash
# From project root
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
pytest src/doom_agent/algorithms/dreamer/v3/tests/
```

## Changelog

For a detailed history of technical improvements (SOLID refactor, Hydra, W&B integration), please refer to the **[CHANGELOG.md](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/CHANGELOG.md)**.

## References

- [Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104)
- [PPO v5 Implementation](link_to_ppo_if_internal)
