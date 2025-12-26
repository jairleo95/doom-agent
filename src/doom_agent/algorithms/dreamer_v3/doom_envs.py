"""
VizDoom environment wrappers for Dreamer V3.

Similar to PPO v5 envs.py but adapted for Dreamer V3's requirements.
"""

import cv2
import numpy as np
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from doom_agent.paths import scenario_path


def preprocess_frame(frame, out_size=(64, 64)):
    """Preprocess VizDoom frame to grayscale normalized image."""
    if frame is None or frame.size == 0:
        return np.zeros((1, out_size[1], out_size[0]), dtype=np.float32)
    
    if frame.ndim == 3 and frame.shape[0] <= 4:
        frame = np.transpose(frame, (1, 2, 0))
    
    # Crop and convert to grayscale
    cropped = frame[40:, 4:-4]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, out_size, interpolation=cv2.INTER_AREA)
    
    # Return uint8 [0, 255] in (H, W, C) format
    return resized[..., None]  # Add channel dimension


def deathmatch_actions():
    """Action list for deathmatch scenario."""
    # [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT]
    return [
        [1, 0, 0, 0, 0, 0, 0],  # Move left
        [0, 1, 0, 0, 0, 0, 0],  # Move right
        [0, 0, 1, 0, 0, 0, 0],  # Attack
        [0, 0, 0, 1, 0, 0, 0],  # Move forward
        [0, 0, 0, 0, 1, 0, 0],  # Move backward
        [0, 0, 0, 0, 0, 1, 0],  # Turn left
        [0, 0, 0, 0, 0, 0, 1],  # Turn right
    ]


def deadly_corridor_actions():
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


def defend_actions():
    """Action list for defend_the_center (TURN_LEFT, TURN_RIGHT, ATTACK)."""
    return [
        [1, 0, 0],  # turn left
        [0, 1, 0],  # turn right
        [0, 0, 1],  # attack
        [0, 0, 0],  # no-op (helps steady aim)
    ]


class DoomDreamerEnv:
    """VizDoom environment wrapper for Dreamer V3."""
    
    def __init__(
        self, 
        scenario='deathmatch.cfg', 
        actions=None,
        frame_skip=4,
        window_visible=False,
        doom_skill=None,
        living_reward=None,
        health_penalty=0.0,
        ammo_penalty=0.0,
        frag_bonus=10.0,
        obs_shape=(64, 64, 1)
    ):
        """
        Args:
            scenario: Path to scenario config file
            actions: List of action vectors
            frame_skip: Number of frames to skip per action
            window_visible: Whether to show game window
            doom_skill: Difficulty level (1-5)
            living_reward: Reward per step (can be negative)
            health_penalty: Penalty per health point lost
            ammo_penalty: Penalty per ammo point used
            frag_bonus: Bonus reward per frag/kill
        """
        self.frame_skip = frame_skip
        self.actions = actions or deathmatch_actions()
        self.health_penalty = health_penalty
        self.ammo_penalty = ammo_penalty
        self.frag_bonus = frag_bonus
        self.obs_shape = obs_shape
        self.out_size = (obs_shape[1], obs_shape[0]) # (W, H) for cv2.resize
        
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
        
        # Track game variables for reward shaping
        self.last_frag_count = 0
        self.last_health = 100.0
        self.last_ammo = 0.0
        
    def reset(self):
        """Reset environment and return initial observation."""
        self.game.new_episode()
        self.last_frag_count = 0
        self.last_health = 100.0
        self.last_ammo = 0.0
        
        state = self.game.get_state()
        if state and len(state.game_variables) >= 3:
            self.last_frag_count = state.game_variables[0]
            self.last_ammo = state.game_variables[1]
            self.last_health = state.game_variables[2]
            
        return preprocess_frame(state.screen_buffer if state else None, self.out_size)
    
    def step(self, action_idx):
        """
        Take a step in the environment.
        
        Args:
            action_idx: Index of action to take
            
        Returns:
            obs: Preprocessed observation
            reward: Reward for this step
            done: Whether episode is finished
        """
        reward = self.game.make_action(self.actions[action_idx], self.frame_skip)
        done = self.game.is_episode_finished()
        
        if not done:
            state = self.game.get_state()
            obs = preprocess_frame(state.screen_buffer, self.out_size)
            
            # Reward shaping
            if state and len(state.game_variables) >= 3:
                # Frag bonus
                frag_count = state.game_variables[0]
                if frag_count > self.last_frag_count:
                    reward += (frag_count - self.last_frag_count) * self.frag_bonus
                self.last_frag_count = frag_count
                
                # Ammo penalty
                ammo = state.game_variables[1]
                if ammo < self.last_ammo:
                    reward -= (self.last_ammo - ammo) * self.ammo_penalty
                self.last_ammo = ammo
                
                # Health penalty
                health = state.game_variables[2]
                if health < self.last_health:
                    reward -= (self.last_health - health) * self.health_penalty
                self.last_health = health
        else:
            obs = np.zeros(self.obs_shape, dtype=np.uint8)
        
        return obs, reward, done
    
    def close(self):
        """Close the environment."""
        self.game.close()
