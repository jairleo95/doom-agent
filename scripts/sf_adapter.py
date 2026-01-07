
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
            self.cfg, self.checkpoint_dict = load_from_checkpoint(model_dir)
            self.cfg.env_frameskip = 4 # Force frameskip matching
            self.actor_critic = create_actor_critic(self.cfg, self.checkpoint_dict['obs_space'], self.checkpoint_dict['action_space'])
            self.actor_critic.load_state_dict(self.checkpoint_dict['model'])
            self.actor_critic.to(self.device)
            self.actor_critic.eval()
            print(f"Sample Factory model loaded from {model_dir}")
            
            # Reset RNN states
            self.rnn_states = torch.zeros([1, self.cfg.hidden_size], dtype=torch.float32, device=self.device)
            
        except Exception as e:
            print(f"❌ Error loading Sample Factory model: {e}")
            self.actor_critic = None

    def select_action(self, obs, health=100, ammo=50, frags=0):
        if self.actor_critic is None:
            return [0]*7

        with torch.no_grad():
            # Prepare observation (SF expects dictionary with 'obs')
            # Normalize and channel-first
            obs_torch = torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            obs_dict = {'obs': obs_torch.to(self.device)}
            
            # Inference
            res = self.actor_critic(obs_dict, self.rnn_states, with_action_distribution=True)
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
            
            # Heuristic mapping (Subject to verification)
            if action_idx == 4: universal[0] = 1 # FWD
            if action_idx == 3: universal[1] = 1 # BWD
            if action_idx == 2: universal[2] = 1 # L
            if action_idx == 1: universal[3] = 1 # R
            if action_idx == 6: universal[4] = 1 # TL
            if action_idx == 5: universal[5] = 1 # TR
            if action_idx == 0: universal[6] = 1 # ATK
            
            return universal

    def reset(self):
        if self.actor_critic:
            self.rnn_states = torch.zeros([1, self.cfg.hidden_size], dtype=torch.float32, device=self.device)
