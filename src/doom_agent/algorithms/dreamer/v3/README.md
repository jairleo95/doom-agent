# Dreamer V3 for Doom

Model-based reinforcement learning implementation using Dreamer V3 for Doom scenarios.

## Architecture

This implementation follows a modular structure similar to PPO v5:

### Files

- **[models.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/nm512_dreamer/models.py)** - Neural network components (Internal)
- **[agent.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/agent.py)** - Main agent class (Adapter)
  - `DreamerV3Agent` - Combines all components
  - Training methods for world model and actor-critic
  - Action selection and state management

- **[replay_buffer.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/replay_buffer.py)** - Experience replay
  - `ReplayBuffer` - Stores transitions and samples sequences
  - **Symmetry Augmentation**: Optimized horizontal flipping for faster convergence.

- **[doom_envs.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/doom_envs.py)** - Environment wrappers (RGB Support)
  - `DoomDreamerEnv` - VizDoom environment wrapper with frame caching
  - High-fidelity metrics collection (frags, health, ammo)

- **[curriculum.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/curriculum.py)** - Training curricula
  - `Stage` - Single training stage configuration
  - `Curriculum` - Multi-stage training curriculum

- **[callbacks.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/callbacks.py)** - Training callbacks
  - `VideoRecorderCallback` - Records high-res episode videos
  - `ImaginationVideoCallback` - Logs world model predictions to TensorBoard
  - `MetricsCallback` - Logs training and detailed gameplay metrics

- **[train.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/train.py)** - Main training script
  - Curriculum-based training loop with Mirror Learning (Symmetry)
  - Stable ETA tracking using EMA

## Usage

### Basic Training

Train on Deathmatch with default curriculum:

```bash
# From project root
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/doom_agent/algorithms/dreamer/v3/train.py --scenario deathmatch --device cuda
```

### Available Scenarios

- `deathmatch` - Deathmatch scenario with progressive difficulty
- `deadly_corridor` - Navigate corridor while fighting enemies
- `defend_the_center` - Defend position against waves
- `universal` - Grand curriculum across multiple scenarios

### Training Options

```bash
python src/doom_agent/algorithms/dreamer/v3/train.py \
  --scenario deathmatch \
  --device cuda \
  --batch-size 128 \
  --batch-length 50 \
  --n-envs 16 \
  --video-freq 50000
```

### Resume Training

```bash
python src/doom_agent/algorithms/dreamer/v3/train.py \
  --scenario deathmatch \
  --resume src/doom_agent/algorithms/dreamer/v3/checkpoints/deathmatch/RUN_ID/skill2_warmup/dreamer_skill2_warmup_final.pt \
  --start-stage 1
```

## Key Features

- **Model-Based (DreamerV3)**: Learns world model for sample efficiency.
- **Symmetry Augmentation**: Mirror learning (Horizontal Flip) to double data efficiency.
- **Imagination Logging**: Visualize agent's "dreams" on TensorBoard.
- **Gameplay Analytics**: Track frags, health remaining, and ammo consumption.
- **RGB Training**: High-fidelity vision for complex environments.
- **Stable ETA**: Precise time estimates using Exponential Moving Average (EMA).

## Output Structure

```
dreamer/v3/
├── tests/              # Unit and integration tests
├── runs/               # TensorBoard logs
│   └── deathmatch/
│       └── RUN_ID/
├── checkpoints/        # Model checkpoints
└── videos/             # Episode GIFs
```

## Testing

Run the test suite from the project root:

```bash
# Unit, Integration and Advanced tests
PYTHONPATH=src python -m unittest src/doom_agent/algorithms/dreamer/v3/tests/test_unit.py \
    src/doom_agent/algorithms/dreamer/v3/tests/test_integration.py \
    src/doom_agent/algorithms/dreamer/v3/tests/test_advanced.py
```

## Comparisons & Theory

| Feature | Dreamer V3 | PPO v5 |
|---------|------------|--------|
| Type | Model-based | Model-free |
| Sample Efficiency | High | Medium |
| Mirror Learning | Yes | Yes |
| Dreaming Log | Yes | No |
| Vision | RGB (64x64) | Grayscale Stack (160x120) |

## Changelog

For a detailed history of technical improvements, bug fixes, and advanced training features, please refer to the **[CHANGELOG.md](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/CHANGELOG.md)**.

## References

- [Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104)
- [Dream to Control: Learning Behaviors by Latent Imagination (DreamerV2)](https://arxiv.org/abs/2010.02193)
