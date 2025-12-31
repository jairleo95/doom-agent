"""Minimal Gymnasium-style VizDoom environments for PPO v4 (SB3)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import cv2
import gymnasium as gym
import numpy as np
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, Button

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


# Universal Buttons (Superset)
UNIVERSAL_BUTTONS = [
    Button.MOVE_FORWARD,
    Button.MOVE_BACKWARD,
    Button.MOVE_LEFT,
    Button.MOVE_RIGHT,
    Button.TURN_LEFT,
    Button.TURN_RIGHT,
    Button.ATTACK
]

def universal_actions() -> list[list[int]]:
    """
    Universal 12-action set based on UNIVERSAL_BUTTONS.
    Indices: FWD(0), BWD(1), L(2), R(3), TL(4), TR(5), ATK(6)
    """
    # 0=FWD, 1=BWD, 2=MV_L, 3=MV_R, 4=TN_L, 5=TN_R, 6=ATK
    return [
        [1, 0, 0, 0, 0, 0, 0],  # 0: Forward
        [0, 0, 0, 0, 0, 0, 1],  # 1: Attack
        [1, 0, 0, 0, 0, 0, 1],  # 2: Forward + Attack
        [0, 0, 0, 0, 1, 0, 0],  # 3: Turn Left
        [0, 0, 0, 0, 0, 1, 0],  # 4: Turn Right
        [0, 0, 0, 0, 1, 0, 1],  # 5: Turn Left + Attack
        [0, 0, 0, 0, 0, 1, 1],  # 6: Turn Right + Attack
        [0, 0, 1, 0, 0, 0, 0],  # 7: Strafe Left
        [0, 0, 0, 1, 0, 0, 0],  # 8: Strafe Right
        [0, 1, 0, 0, 0, 0, 0],  # 9: Backward
        [1, 0, 0, 0, 1, 0, 0],  # 10: Forward + Turn Left
        [1, 0, 0, 0, 0, 1, 0],  # 11: Forward + Turn Right
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

        # Universal Action Mapping Logic
        # We need to map self.actions (which are vectors of length len(UNIVERSAL_BUTTONS))
        # to the game's actual buttons.
        # If self.actions is provided explicitly as "old style" (matching scenario buttons),
        # we skip mapping. But for PPO v5 Curriculum, we pass universal_actions().
        
        game_buttons = self.game.get_available_buttons()
        self.button_map = []
        is_universal = len(self.actions[0]) == len(UNIVERSAL_BUTTONS)
        
        # Heuristic: If action len matches universal len, assume universal.
        # Ideally we should pass a flag, but this works if lengths differ from old scenarios.
        # deadly_corridor has 7 buttons, same as universal. But order might differ?
        # deadly_corridor order: L, R, ATK, FWD, BWD, TL, TR.
        # Universal order: FWD, BWD, L, R, TL, TR, ATK.
        # So we MUST map if we use universal_actions().
        
        # Let's assume if the action set passed IS universal_actions(), then we map.
        # We can just always compute map. 
        # But we need to know what the input vector represents.
        # We will assume input vector corresponds to UNIVERSAL_BUTTONS order IFF 
        # we are in v5 universal mode. 
        # For safety, let's assume we ALWAYS map if the input actions are intended to be Universal.
        # Since I am controlling the call in train.py, I will pass universal_actions().
        
        self.use_mapping = False
        if len(self.actions[0]) == len(UNIVERSAL_BUTTONS):
             # Check if we should enable mapping. 
             # Let's just do it dynamically: 
             # Map index i of input vector (UNIVERSAL_BUTTONS[i]) 
             # to index j of game_buttons where game_buttons[j] == UNIVERSAL_BUTTONS[i].
             self.mapping_indices = []
             for btn in UNIVERSAL_BUTTONS:
                 try:
                     self.mapping_indices.append(game_buttons.index(btn))
                 except ValueError:
                     self.mapping_indices.append(-1) # Button not available in this scenario
             self.use_mapping = True
        
        h, w = frame_size[1], frame_size[0]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)
        self.state = np.zeros((h, w, 1), dtype=np.uint8)
        self.last_killcount = 0
        self.last_health = 100.0
        self.last_ammo = 0.0  # Will be updated on first step or reset

    def step(self, action: int):
        if self.use_mapping:
            # Map universal vector to scenario vector
            univ_vector = self.actions[action]
            # VizDoom make_action needs a list of ints.
            # But wait, make_action expects a list of length equal to available_buttons.
            available_buttons_count = len(self.game.get_available_buttons())
            scenario_action = [0] * available_buttons_count
            
            for i, target_idx in enumerate(self.mapping_indices):
                if target_idx != -1:
                    scenario_action[target_idx] = univ_vector[i]
            
            reward = self.game.make_action(scenario_action, self.frame_skip)
        else:
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
