"""
VizDoom environment wrappers for Dreamer V3.

Similar to PPO v5 envs.py but adapted for Dreamer V3's requirements.
"""

import cv2
import numpy as np
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, Button, GameVariable

from doom_agent.paths import scenario_path


def robust_transpose(frame):
    """Ensure frame is in (H, W, C) format."""
    if frame is None or frame.ndim < 2:
        return frame
    
    # If it's (C, H, W) - VizDoom default
    if frame.ndim == 3 and frame.shape[0] <= 4:
        return np.transpose(frame, (1, 2, 0))
    
    # Already (H, W, C) or (H, W)
    return frame


def preprocess_frame(frame, out_size=(64, 64), color=False):
    """Preprocess VizDoom frame to normalized image."""
    frame = robust_transpose(frame)
    if frame is None or frame.size == 0:
        channels = 3 if color else 1
        return np.zeros((out_size[1], out_size[0], channels), dtype=np.uint8)
    
    if not color:
        if frame.ndim == 3 and frame.shape[-1] > 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if frame.ndim == 2:
            frame = frame[..., None]
    
    resized = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
    
    if resized.ndim == 2:
        resized = resized[..., None]
        
    return resized.astype(np.uint8)


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

def universal_actions():
    """
    Universal 12-action set based on UNIVERSAL_BUTTONS.
    Indices: FWD(0), BWD(1), L(2), R(3), TL(4), TR(5), ATK(6)
    """
    # 0=FWD, 1=BWD, 2=MV_L, 3=MV_R, 4=TN_L, 5=TN_R, 6=ATK
    return [
        [1, 0, 0, 0, 0, 0, 0],  # 0: forward
        [0, 0, 0, 0, 0, 0, 1],  # 1: attack
        [1, 0, 0, 0, 0, 0, 1],  # 2: forward + attack
        [0, 0, 0, 0, 1, 0, 0],  # 3: turn left
        [0, 0, 0, 0, 0, 1, 0],  # 4: turn right
        [0, 0, 0, 0, 1, 0, 1],  # 5: turn left + attack
        [0, 0, 0, 0, 0, 1, 1],  # 6: turn right + attack
        [0, 0, 1, 0, 0, 0, 0],  # 7: strafe left
        [0, 0, 0, 1, 0, 0, 0],  # 8: strafe right
        [0, 1, 0, 0, 0, 0, 0],  # 9: backward
        [1, 0, 0, 0, 1, 0, 0],  # 10: forward + turn left
        [1, 0, 0, 0, 0, 1, 0],  # 11: forward + turn right
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
        movement_reward=0.0,
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
        self.movement_reward = movement_reward
        self.obs_shape = obs_shape
        self.color = (obs_shape[-1] == 3)
        self.out_size = (obs_shape[1], obs_shape[0]) # (W, H) for cv2.resize
        
        # Cache for high-resolution video recording
        self.last_high_res_render = None
        
        self.game = DoomGame()
        self.game.load_config(scenario_path(scenario))
        self.game.set_mode(Mode.PLAYER)
        self.game.set_sound_enabled(False)
        
        if doom_skill is not None:
            self.game.set_doom_skill(int(doom_skill))
        if living_reward is not None:
            self.game.set_living_reward(float(living_reward))
            
        # Add position variables for movement reward if needed
        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_window_visible(window_visible)
        self.game.init()
        
        # Universal Action Mapping Logic
        game_buttons = self.game.get_available_buttons()
        self.button_map = []
        is_universal = self.actions and len(self.actions[0]) == len(UNIVERSAL_BUTTONS)
        
        if is_universal:
            # Create a map from UNIVERSAL_BUTTONS to current scenario buttons
            for b in UNIVERSAL_BUTTONS:
                if b in game_buttons:
                    self.button_map.append(game_buttons.index(b))
                else:
                    self.button_map.append(None)
        
        # Track game variables for reward shaping
        self.last_frag_count = 0
        self.last_health = 100.0
        self.last_ammo = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        
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
            
            # Position variables are added after the standard ones
            if len(state.game_variables) >= 5:
                self.last_x = state.game_variables[3]
                self.last_y = state.game_variables[4]
            
        # Cache color high-res version for video recorder
        if state and state.screen_buffer is not None:
            self.last_high_res_render = robust_transpose(state.screen_buffer)
        
        return preprocess_frame(state.screen_buffer if state else None, self.out_size, color=self.color)
    
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
        action_vec = self.actions[action_idx]
        
        # If we have a button map, we need to translate the universal action vector
        # to the scenario's specific button vector.
        if self.button_map:
            actual_action = [0] * len(self.game.get_available_buttons())
            for i, mapped_idx in enumerate(self.button_map):
                if mapped_idx is not None and action_vec[i] == 1:
                    actual_action[mapped_idx] = 1
            action_vec = actual_action
            
        reward = self.game.make_action(action_vec, self.frame_skip)
        done = self.game.is_episode_finished()
        
        if not done:
            state = self.game.get_state()
            
            # Cache color high-res version for video recorder
            if state and state.screen_buffer is not None:
                self.last_high_res_render = robust_transpose(state.screen_buffer)
            
            # Update gameplay variables for metrics
            if state and len(state.game_variables) >= 3:
                self.last_frag_count = state.game_variables[0]
                self.last_ammo = state.game_variables[1]
                self.last_health = state.game_variables[2]
            
            obs = preprocess_frame(state.screen_buffer, self.out_size, color=self.color)
            
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

                # Movement reward (Anti-camping)
                if self.movement_reward > 0 and len(state.game_variables) >= 5:
                    x = state.game_variables[3]
                    y = state.game_variables[4]
                    dx = x - self.last_x
                    dy = y - self.last_y
                    dist = np.sqrt(dx**2 + dy**2)
                    reward += dist * self.movement_reward
                    self.last_x = x
                    self.last_y = y
        else:
            obs = np.zeros(self.obs_shape, dtype=np.uint8)
        
        return obs, reward, done
    
    def close(self):
        """Close the environment."""
        self.game.close()
