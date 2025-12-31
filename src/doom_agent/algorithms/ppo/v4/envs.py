"""Minimal Gymnasium-style VizDoom environments for PPO v4 (SB3)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import cv2
import gymnasium as gym
import numpy as np
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from doom_agent.paths import scenario_path


def deadly_corridor_actions() -> list[list[int]]:
    """Action list for deadly_corridor scenario."""
    # [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT]
    return [
        [0, 0, 0, 1, 0, 0, 0],  # forward
        [0, 0, 1, 0, 0, 0, 0],  # attack
        [0, 0, 1, 1, 0, 0, 0],  # forward + attack
        [0, 0, 0, 0, 0, 1, 0],  # turn left
        [0, 0, 0, 0, 0, 0, 1],  # turn right
        [0, 0, 1, 0, 0, 1, 0],  # turn left + attack
        [0, 0, 1, 0, 0, 0, 1],  # turn right + attack
        [1, 0, 0, 0, 0, 0, 0],  # strafe left
        [0, 1, 0, 0, 0, 0, 0],  # strafe right
        [0, 0, 0, 0, 1, 0, 0],  # backward
        [0, 0, 0, 1, 0, 1, 0],  # forward + turn left
        [0, 0, 0, 1, 0, 0, 1],  # forward + turn right
    ]


def defend_actions() -> list[list[int]]:
    """Action list for defend_the_center (TURN_LEFT, TURN_RIGHT, ATTACK)."""
    return [
        [1, 0, 0],  # turn left
        [0, 1, 0],  # turn right
        [0, 0, 1],  # attack
        [0, 0, 0],  # no-op (helps steady aim)
    ]


def preprocess_frame(frame: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Convert VizDoom frame to grayscale HxWx1 uint8."""
    if frame is None or frame.size == 0:
        return np.zeros((out_size[1], out_size[0], 1), dtype=np.uint8)

    if frame.ndim == 3 and frame.shape[0] <= 4:  # (C,H,W) -> (H,W,C)
        frame = np.transpose(frame, (1, 2, 0))

    cropped = frame[40:, 4:-4]  # remove HUD borders
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, out_size, interpolation=cv2.INTER_AREA)
    return resized[..., None].astype(np.uint8)


class DoomCorridorEnv(gym.Env):
    """Simple Gymnasium-compatible VizDoom env for deadly_corridor."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "deadly_corridor.cfg",
        frame_skip: int = 4,
        frame_size: Tuple[int, int] = (160, 120),
        actions: list[list[int]] | None = None,
        frame_processor: Callable[[np.ndarray], np.ndarray] | None = None,
        window_visible: bool = False,
        doom_skill: int | None = None,
        living_reward: float | None = None,
        health_penalty: float = 0.0,
        ammo_penalty: float = 0.0,
    ):
        super().__init__()
        self.frame_skip = frame_skip
        self.frame_size = frame_size
        self.actions = actions or default_actions()
        self.frame_processor = frame_processor or (lambda f: preprocess_frame(f, self.frame_size))
        self.health_penalty = health_penalty
        self.ammo_penalty = ammo_penalty

        self.game = DoomGame()
        self.game.load_config(scenario_path(scenario))
        self.game.set_mode(Mode.PLAYER)
        if doom_skill is not None:
            self.game.set_doom_skill(int(doom_skill))
        if living_reward is not None:
            self.game.set_living_reward(float(living_reward))
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_window_visible(window_visible)
        self.game.init()

        h, w = frame_size[1], frame_size[0]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
        self.state = np.zeros((h, w, 1), dtype=np.uint8)
        self.last_killcount = 0
        self.last_health = 100.0
        self.last_ammo = 0.0  # Will be updated on first step or reset

    def step(self, action: int):
        reward = self.game.make_action(self.actions[action], self.frame_skip)
        terminated = self.game.is_episode_finished()

        if not terminated:
            state = self.game.get_state()
            self.state = self.frame_processor(state.screen_buffer if state else None)
            
            # Reward Shaping:
            # available_game_variables = { KILLCOUNT AMMO2 HEALTH }
            if state and len(state.game_variables) >= 3:
                # 1. Kill Bonus
                killcount = state.game_variables[0]
                if killcount > self.last_killcount:
                    reward += (killcount - self.last_killcount) * 1.0
                self.last_killcount = killcount

                # 2. Ammo Penalty (game_variables[1] is AMMO2)
                ammo = state.game_variables[1]
                if ammo < self.last_ammo:
                     # Penalize ammo USE (diff expected to be positive if used)
                     reward -= (self.last_ammo - ammo) * self.ammo_penalty
                self.last_ammo = ammo

                # 3. Health Penalty (game_variables[2] is HEALTH)
                health = state.game_variables[2]
                if health < self.last_health:
                    reward -= (self.last_health - health) * self.health_penalty
                self.last_health = health

        else:
            self.state = np.zeros_like(self.state)

        return self.state, reward, terminated, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        self.last_killcount = 0
        self.last_health = 100.0
        self.last_ammo = 0.0
        state = self.game.get_state()
        if state and len(state.game_variables) >= 3:
            self.last_ammo = state.game_variables[1]
            self.last_health = state.game_variables[2]
        self.state = self.frame_processor(state.screen_buffer if state else None)
        return self.state, {}

    def close(self):
        self.game.close()


def make_vec_env(n_envs: int = 8, **env_kwargs):
    """Create vectorized environment for SB3."""
    return gym.vector.SyncVectorEnv([lambda: DoomCorridorEnv(**env_kwargs) for _ in range(n_envs)])
