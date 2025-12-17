import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "deadly_corridor"
RUNS_DIR = BASE_DIR / "runs" / "vizdoom_ppo_deadly_corridor"


SEED = 123
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# acelera kernels en GPU sin degradar calidad notable
torch.set_float32_matmul_precision("high")
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True


def set_global_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class StageConfig:
    name: str
    doom_skill: int
    living_reward: float
    reward_scale: float
    min_episodes: int
    unlock_mean_reward: float
    window: int


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.1
    lr: float = 2e-4  # sube un poco para converger con menos updates
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    ppo_epochs: int = 2  # menos epochs por update acelera
    mini_batch_size: int = 512  # menos mini-batches por update
    entropy_start: float = 0.02
    entropy_end: float = 0.001
    rollout_steps: int = 2048  # rollouts más cortos → updates más frecuentes
    num_updates: int = 600
    use_amp: bool = True


@dataclass(frozen=True)
class CheckpointConfig:
    directory: str = str(CHECKPOINT_DIR)
    interval: int = 25
    best_name: str = "best.pth"
    periodic_template: str = "update_{:04d}.pth"


@dataclass(frozen=True)
class EnvConfig:
    n_envs: int = 10
    frame_stack: int = 4
    img_size: tuple = (84, 84)


@dataclass(frozen=True)
class ShapingConfig:
    reward_kill: float = 1.0
    reward_health_scale: float = 0.01
    reward_ammo_scale: float = 0.005
    attack_cooldown: int = 2
    attack_spam_penalty: float = 0.01


PPO_CFG = PPOConfig()
CKPT_CFG = CheckpointConfig()
ENV_CFG = EnvConfig()
SHAPING_CFG = ShapingConfig()

CURRICULUM_STAGES = (
    StageConfig(
        name="skill2_warmup",
        doom_skill=2,
        living_reward=-0.01,
        reward_scale=1.0 / 30.0,
        min_episodes=6,
        unlock_mean_reward=-15.0,
        window=4,
    ),
    StageConfig(
        name="skill4_standard",
        doom_skill=4,
        living_reward=0.0,
        reward_scale=1.0 / 40.0,
        min_episodes=8,
        unlock_mean_reward=40.0,
        window=6,
    ),
    StageConfig(
        name="skill5_target",
        doom_skill=5,
        living_reward=0.0,
        reward_scale=1.0 / 50.0,
        min_episodes=12,
        unlock_mean_reward=75.0,
        window=8,
    ),
)

# convenience aliases
FRAME_STACK = ENV_CFG.frame_stack
IMG_SIZE = ENV_CFG.img_size
