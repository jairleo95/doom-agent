
import sys
import torch
import numpy as np
from pathlib import Path
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.cfg.arguments import load_from_checkpoint

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

class SampleFactoryAdapter:
    def __init__(self, model_dir):
        self.device = torch.device('cpu')
        
        # Load config and model
        try:
            # SF 2.x expect arguments object
            mock_cfg = AttrDict()
            mock_cfg.train_dir = str(Path(model_dir).parent)
            mock_cfg.experiment = Path(model_dir).name
            
            # Since we downloaded files into a flat directory 'external/sample_factory_model'
            # we need to trick SF into finding config.json there.
            # load_from_checkpoint looks for train_dir/experiment/config.json
            
            # Better approach: Load config manually and create actor critic manually
            import json
            config_path = Path(model_dir) / "config.json"
            with open(config_path, "r") as f:
                json_params = json.load(f)
            
            self.cfg = AttrDict(json_params)
            self.cfg.env_frameskip = 4
            if not hasattr(self.cfg, 'hidden_size'):
                self.cfg.hidden_size = 512
            # SF uses 'rnn_size' in core.py, often aliased or separate
            if not hasattr(self.cfg, 'rnn_size'):
                self.cfg.rnn_size = self.cfg.hidden_size
            
            # Load checkpoint manually
            checkpoint_path = Path(model_dir) / "checkpoint.pth"
            # PyTorch 2.6+ defaults weights_only=True which breaks legacy checkpoints with numpy scalars
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Manually construct spaces if missing from checkpoint (Common in some SF versions)
            import gymnasium as gym
            # Standard SampleFactory Doom env shape usually (3, H, W). Checkpoint says (3, 72, 128)
            obs_box = gym.spaces.Box(low=0, high=255, shape=(3, 72, 128), dtype=np.uint8)
            # Sample Factory expects a Dict space with "obs" key
            obs_space = gym.spaces.Dict({"obs": obs_box})
            
            # Checkpoint output layer is size 39. This implies 39 discrete actions/buttons combos.
            action_space = gym.spaces.Discrete(39)

            self.actor_critic = create_actor_critic(self.cfg, obs_space, action_space)
            
            # --- COMPATIBILITY FIX: REMAP KEYS ---
            sd = checkpoint['model']
            new_sd = {}
            for k, v in sd.items():
                new_k = k
                # Convert basic_encoder to encoders.obs
                if "encoder.basic_encoder" in k:
                    new_k = k.replace("encoder.basic_encoder", "encoder.encoders.obs")
                
                if "measurements" in k:
                    continue
                    
                new_sd[new_k] = v
                
            self.actor_critic.load_state_dict(new_sd, strict=False)
            self.actor_critic.to(self.device)
            self.actor_critic.eval()
            print(f"Sample Factory model loaded from {model_dir}")
            
            # Reset RNN states (For LSTM, size is 2 * hidden_size for h and c)
            self.rnn_states = torch.zeros([1, 2 * self.cfg.hidden_size], dtype=torch.float32, device=self.device)
            
        except Exception as e:
            print(f"❌ Error loading Sample Factory model: {e}")
            import traceback
            traceback.print_exc()
            self.actor_critic = None
            sys.exit(1) # Fail hard for debugging

    def select_action(self, obs, health=100, ammo=50, frags=0):
        if self.actor_critic is None:
            return [0]*7

        with torch.no_grad():
            # Prepare observation (SF expects dictionary with 'obs')
            # Normalize and channel-first
            
            # RESIZE to match model expectation (128x72) <-> Box(3, 72, 128)
            import cv2
            obs_resized = cv2.resize(obs, (128, 72), interpolation=cv2.INTER_AREA)
            
            obs_torch = torch.from_numpy(obs_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            obs_dict = {'obs': obs_torch.to(self.device)}
            
            # Inference
            res = self.actor_critic(obs_dict, self.rnn_states)
            self.rnn_states = res['new_rnn_states']
            
            # Action Translation (SampleFactory assumes standard VizDoom env actions usually)
            # This specific model uses a discrete action space of size 8 for DefendCenter/Deathmatch?
            # We need to map index to button.
            
            action_idx = res['actions'].item()
            
            # VizDoom default available_actions often map to:
            # [TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT, ATTACK, SPEED]
            # BUT: SampleFactory models often reshape this. 
            
            # Mapping based on common SF VizDoom configs:
            # 0: ATTACK
            # 1: MOVE_RIGHT
            # 2: MOVE_LEFT
            # 3: MOVE_BACKWARD
            # 4: MOVE_FORWARD
            # 5: TURN_RIGHT
            # 6: TURN_LEFT
            # 7: SPEED
            
            # Universal: [FWD, BWD, L, R, TL, TR, ATK]
            universal = [0]*7
            
            # 39 actions typically imply a composite action space flattened.
            # Without the actions enum from the original training code, we can only guess.
            # Standard VizDoom usually has MOVE_FORWARD as index 0 or near it.
            
            # Heuristic mapping for validation
            # Assuming typical key layouts or single-button presses are lower indices
            if action_idx == 0: universal[6] = 1 # Attack
            if action_idx == 1: universal[0] = 1 # Fwd
            if action_idx == 2: universal[1] = 1 # Bwd
            if action_idx == 3: universal[2] = 1 # L
            if action_idx == 4: universal[3] = 1 # R
            if action_idx == 5: universal[4] = 1 # TL
            if action_idx == 6: universal[5] = 1 # TR
            
            return universal

    def reset(self):
        if self.actor_critic:
            self.rnn_states = torch.zeros([1, self.cfg.hidden_size], dtype=torch.float32, device=self.device)
