import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

# Add external/arnold to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "external/arnold"))

# Arnold imports
try:
    from src.model import get_model_class
    from src.doom.actions import ActionBuilder
    from src.utils import get_device_mapping
except ImportError:
    print("❌ ERROR: No se pudieron cargar los módulos de Arnold. Asegúrate de que external/arnold existe.")
    sys.exit(1)

class Params:
    """Mock params object for Arnold."""
    def __init__(self):
        self.network_type = "dqn_rnn"
        self.recurrence = "lstm"
        self.n_rec_layers = 1
        self.hidden_dim = 512
        self.dropout = 0.5
        self.use_bn = False
        self.clip_delta = 1.0
        self.dueling_network = False
        self.n_fm = 3
        self.height = 60
        self.width = 108
        self.hist_size = 4
        self.remember = True
        self.game_features = "target,enemy"
        self.n_features = 2 # target, enemy
        self.game_variables = [('health', 101), ('sel_ammo', 301)]
        self.n_variables = 2
        self.variable_dim = [32, 32]
        self.bucket_size = [10, 1]
        self.action_combinations = "move_fb+move_lr;turn_lr;attack"
        self.use_continuous = False
        self.speed = "on"
        self.crouch = "off"
        self.freelook = False
        self.gpu_id = -1 # Run on CPU by default for compatibility
        self.batch_size = 1 # For inference

class ArnoldAdapter:
    def __init__(self, model_path):
        self.params = Params()
        
        # 1. Action Builder
        self.action_builder = ActionBuilder(self.params)
        
        # 2. Network Initialization
        self.network = get_model_class(self.params.network_type)(self.params)
        
        # 3. Load Model
        print(f"Loading Arnold model from {model_path}...")
        map_location = get_device_mapping(self.params.gpu_id)
        reloaded = torch.load(model_path, map_location=map_location)
        self.network.module.load_state_dict(reloaded)
        self.network.module.eval()
        
        # State tracking
        self.reset()

    def reset(self):
        self.network.reset()
        self.last_states = []

    def preprocess_obs(self, obs_rgb):
        """Resize and transpose observation for Arnold."""
        # Standard input is usually (H, W, 3)
        resized = cv2.resize(obs_rgb, (self.params.width, self.params.height))
        # Arnold expects (CH, H, W)
        return resized.transpose(2, 0, 1)

    def select_action(self, obs_rgb, health=100, ammo=50, frags=0):
        """
        Select action based on observation and game variables.
        returns: Universal action vector [FWD, BWD, L, R, TL, TR, ATK]
        """
        screen = self.preprocess_obs(obs_rgb)
        variables = np.array([health, ammo], dtype=np.int64)
        
        class State:
            def __init__(self, s, v):
                self.screen = s
                self.variables = v
        
        state = State(screen, variables)
        
        # Initialize history with the first state if empty
        if not self.last_states:
            for _ in range(self.params.hist_size):
                self.last_states.append(state)
        else:
            self.last_states.append(state)
            # Maintain history size
            if len(self.last_states) > self.params.hist_size:
                self.last_states.pop(0)
            
        # Get action index from Arnold
        action_id = self.network.next_action(self.last_states)
        
        # Translate Arnold action to Universal format
        return self.translate_action(action_id)

    def translate_action(self, action_id):
        """
        Arnold actions are subsets. We need to convert its output list to our 7-button vector.
        Arnold internal: [MF, MB, TL, TR, ML, MR, ATK, SPEED, CROUCH]
        Universal: [FWD, BWD, L, R, TL, TR, ATK]
        """
        # Get Arnold's boolean action vector
        arnold_vec = self.action_builder.get_action(action_id)
        
        # Arnold indices (based on get_available_buttons):
        # 0: MOVE_FORWARD
        # 1: MOVE_BACKWARD
        # 2: TURN_LEFT
        # 3: TURN_RIGHT
        # 4: MOVE_LEFT
        # 5: MOVE_RIGHT
        # 6: ATTACK
        # 7: SPEED (Always on)
        # 8: CROUCH (Always off)
        
        # Note: action_builder.get_action returns a list of booleans/ints
        
        universal_vec = [0] * 7
        if arnold_vec[0]: universal_vec[0] = 1 # FWD
        if arnold_vec[1]: universal_vec[1] = 1 # BWD
        if arnold_vec[4]: universal_vec[2] = 1 # L
        if arnold_vec[5]: universal_vec[3] = 1 # R
        if arnold_vec[2]: universal_vec[4] = 1 # TL
        if arnold_vec[3]: universal_vec[5] = 1 # TR
        if arnold_vec[6]: universal_vec[6] = 1 # ATK
        
        return universal_vec
