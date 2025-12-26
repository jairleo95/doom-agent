# Dreamer V3 for Doom

Model-based reinforcement learning implementation using Dreamer V3 for Doom scenarios.

## Architecture

This implementation follows a modular structure similar to PPO v5:

### Files

- **[models.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer_v3/models.py)** - Neural network components
  - `Encoder` - CNN for encoding observations
  - `Decoder` - CNN for reconstructing observations
  - `RSSM` - Recurrent State-Space Model (world model)
  - `RewardPredictor` - Predicts rewards from latent states
  - `ContinuePredictor` - Predicts episode continuation
  - `Actor` - Policy network
  - `Critic` - Value network

- **[agent.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer_v3/agent.py)** - Main agent class
  - `DreamerV3Agent` - Combines all components
  - Training methods for world model and actor-critic
  - Action selection and state management

- **[replay_buffer.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer_v3/replay_buffer.py)** - Experience replay
  - `ReplayBuffer` - Stores transitions and samples sequences

- **[envs.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer_v3/envs.py)** - Environment wrappers
  - `DoomDreamerEnv` - VizDoom environment wrapper
  - Action definitions for different scenarios
  - Frame preprocessing

- **[curriculum.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer_v3/curriculum.py)** - Training curricula
  - `Stage` - Single training stage configuration
  - `Curriculum` - Multi-stage training curriculum
  - Predefined curricula for different scenarios

- **[callbacks.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer_v3/callbacks.py)** - Training callbacks
  - `VideoRecorderCallback` - Records episode videos
  - `CheckpointCallback` - Saves model checkpoints
  - `EvalCallback` - Periodic evaluation
  - `MetricsCallback` - Logs training metrics

- **[train.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer_v3/train.py)** - Main training script
  - Curriculum-based training loop
  - Command-line interface
  - Experiment tracking

## Usage

### Basic Training

Train on Deathmatch with default curriculum:

```bash
cd src/doom_agent/algorithms/dreamer_v3
python train.py --scenario deathmatch --device cuda
```

### Available Scenarios

- `deathmatch` - Deathmatch scenario with progressive difficulty
- `deadly_corridor` - Navigate corridor while fighting enemies
- `defend_the_center` - Defend position against waves
- `universal` - Grand curriculum across multiple scenarios

### Training Options

```bash
python train.py \
  --scenario deathmatch \
  --device cuda \
  --batch-size 32 \
  --sequence-length 50 \
  --imagination-horizon 20 \
  --save-every 100 \
  --eval-every 50 \
  --video-every 100
```

### Resume Training

```bash
python train.py \
  --scenario deathmatch \
  --resume checkpoints/deathmatch/20250126-015000/skill2_warmup/dreamer_skill2_warmup_ep500.pt \
  --start-stage 1
```

## Curriculum Training

Each scenario has a multi-stage curriculum that progressively increases difficulty:

### Deathmatch Curriculum

1. **Skill 2 Warmup** (500 episodes) - Learn basic combat
2. **Skill 3 Intermediate** (1000 episodes) - Improve tactics
3. **Skill 4 Advanced** (1500 episodes) - Master combat
4. **Skill 5 Expert** (2000 episodes) - Ultimate challenge

### Grand Curriculum

1. **Basic** - Movement and shooting
2. **Defend Center** - Defense and ammo management
3. **Deadly Corridor** - Navigation and combat
4. **Deathmatch** - Full deathmatch

## How It Works

### World Model Learning

1. **Collect Experience**: Agent interacts with environment
2. **Store in Buffer**: Transitions stored in replay buffer
3. **Sample Sequences**: Sample sequences of length 50
4. **Train World Model**:
   - Encoder learns to compress observations
   - RSSM learns environment dynamics
   - Decoder reconstructs observations
   - Reward predictor learns reward function
   - Continue predictor learns termination

### Policy Learning

1. **Imagine Trajectories**: Use world model to imagine future
2. **Compute Returns**: Calculate λ-returns from imagined rewards
3. **Train Critic**: Predict values of imagined states
4. **Train Actor**: Improve policy using advantages

## Key Features

- **Model-Based**: Learns world model for sample efficiency
- **Curriculum Learning**: Progressive difficulty stages
- **Modular Design**: Clean separation of concerns
- **Comprehensive Logging**: Metrics, videos, checkpoints
- **Resume Support**: Continue training from checkpoints

## Hyperparameters

### Model Architecture
- Embedding: 1024 dims
- Hidden state: 512 dims
- Stochastic state: 32×32 discrete categorical

### Training
- World model LR: 3e-4
- Actor LR: 8e-5
- Critic LR: 8e-5
- Batch size: 16
- Sequence length: 50
- Imagination horizon: 15
- Gamma: 0.99
- Lambda: 0.95

## Output Structure

```
dreamer_v3/
├── runs/
│   └── deathmatch/
│       └── 20250126-015000/
│           ├── config.json
│           └── skill2_warmup/
│               └── metrics.json
├── checkpoints/
│   └── deathmatch/
│       └── 20250126-015000/
│           ├── skill2_warmup/
│           │   ├── dreamer_skill2_warmup_ep100.pt
│           │   └── dreamer_skill2_warmup_final.pt
│           └── videos/
│               └── skill2_warmup/
│                   └── dreamer_skill2_warmup_ep100.gif
```

## Comparison with PPO v5

| Feature | Dreamer V3 | PPO v5 |
|---------|------------|--------|
| Type | Model-based | Model-free |
| Sample Efficiency | High | Medium |
| Computation | High | Low |
| Memory | High (replay buffer) | Low |
| Curriculum | Episode-based | Timestep-based |
| Training | World model + Actor-Critic | Actor-Critic only |

## References

- [Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104)
- [Dream to Control: Learning Behaviors by Latent Imagination (DreamerV2)](https://arxiv.org/abs/2010.02193)
