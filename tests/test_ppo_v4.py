
import pytest
import numpy as np
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from doom_agent.algorithms.ppo_v4.train_deadly_corridor import build_envs as build_corridor, Stage
from doom_agent.algorithms.ppo_v4.train_defend_the_center import build_envs as build_defend
from stable_baselines3 import PPO

@pytest.fixture
def corridor_stage():
    return Stage(name="test_corridor", timesteps=100, doom_skill=1, living_reward=0.0, frame_skip=4)

@pytest.fixture
def defend_stage():
    return Stage(name="test_defend", timesteps=100, doom_skill=1, living_reward=0.0, frame_skip=4)

def test_corridor_env_builder(corridor_stage):
    """Test that deadly_corridor env is built with correct shape and wrappers."""
    env = build_corridor(n_envs=2, stage=corridor_stage, window_visible=False)
    try:
        obs = env.reset()
        # Verify shape: (n_envs, n_stack, height, width)
        assert obs.shape == (2, 4, 120, 160)
        assert obs.dtype == np.uint8
        # Verify action space (discrete)
        assert env.action_space.n > 0
    finally:
        env.close()

def test_defend_env_builder(defend_stage):
    """Test that defend_the_center env is built with correct shape and wrappers."""
    env = build_defend(n_envs=2, stage=defend_stage, window_visible=False)
    try:
        obs = env.reset()
        assert obs.shape == (2, 4, 120, 160)
    finally:
        env.close()

def test_ppo_training_smoke(corridor_stage):
    """Smoke test: Verify PPO can perform a few updates without crashing."""
    env = build_corridor(n_envs=2, stage=corridor_stage, window_visible=False)
    try:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            n_steps=16, # Small for testing
            batch_size=4,
            verbose=1
        )
        # Learn for a very short duration
        model.learn(total_timesteps=50)
    finally:
        env.close()

def test_action_spaces_distinct(corridor_stage, defend_stage):
    """Verify that different scenarios have appropriate action spaces."""
    env_c = build_corridor(n_envs=1, stage=corridor_stage)
    env_d = build_defend(n_envs=1, stage=defend_stage)
    try:
        # deadly_corridor usually has more actions (move + turn + strafe etc)
        # defend_the_center usually has fewer (turn + attack)
        assert env_c.action_space.n != env_d.action_space.n
    finally:
        env_c.close()
        env_d.close()
