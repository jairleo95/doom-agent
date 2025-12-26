"""
Dreamer V3 Agent Adapter for NM512 implementation.

Wraps the NM512/dreamerv3-torch implementation to fit our interface.
"""

import sys
import pathlib
import torch
import numpy as np
import ruamel.yaml as yaml

# Add NM512 repo to path (prepend to ensure its imports like 'envs' take precedence)
nm512_path = pathlib.Path(__file__).parent / "nm512_dreamer"
sys.path.insert(0, str(nm512_path))

try:
    from nm512_dreamer import tools
    from nm512_dreamer import dreamer
    from nm512_dreamer import exploration as expl
except ImportError:
    print("Error: Could not import NM512 Dreamer implementation.")
    print(f"Make sure {nm512_path} exists and dependencies are installed.")
    raise


class DoomObsSpace:
    def __init__(self, shape):
        self.shape = shape
    
    def __repr__(self):
        return f"Box({self.shape}, uint8)"


class DoomActSpace:
    def __init__(self, n):
        self.n = n
        self.shape = (n,)
        self.discrete = True
    
    def __repr__(self):
        return f"Discrete({self.n})"

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class DoomDictObsSpace:
    def __init__(self, spaces):
        self.spaces = spaces
    
    def __repr__(self):
        return f"Dict({self.spaces})"

class DreamerV3Agent:
    """Adapter for NM512 Dreamer V3 agent."""
    
    def __init__(self, config, run_dir):
        """
        Args:
            config: Dictionary with configuration parameters (merged with defaults)
            run_dir: Path to run directory for logging
        """
        self.run_dir = pathlib.Path(run_dir)
        self.device = torch.device(config['device'])
        
        # Load default config from NM512 repo
        default_config_path = nm512_path / "configs.yaml"
        yaml_parser = yaml.YAML(typ='safe', pure=True)
        defaults = yaml_parser.load(default_config_path.read_text())['defaults']
        
        # Merge our config into defaults recursively
        self.config = AttributeDict(defaults)
        self._recursive_update(self.config, config)
        
        # Ensure recursive AttributeDict for nested dicts
        self._ensure_attribute_dict(self.config)

        # Ensure critical configs match Doom
        self.config.logdir = str(self.run_dir)
        self.config.traindir = str(self.run_dir / "train_eps")
        self.config.evaldir = str(self.run_dir / "eval_eps")
        
        # Initialize Logger
        # Start at step=action_repeat to prevent Dreamer from training immediately at step 0 (0 % large_num == 0)
        # step // action_repeat becomes 1.
        self.logger = tools.Logger(self.run_dir, self.config.action_repeat)
        
        # Define Spaces
        # Obs: (C, H, W) -> NM512 expects dict with keys. 
        # We will wrap our obs in a Dict space structure.
        self.obs_space = DoomDictObsSpace({'image': DoomObsSpace(self.config.obs_shape)}) 
        self.act_space = DoomActSpace(self.config.action_dim)
        
        # Dataset
        # NM512 Dreamer expects a dataset generator in __init__.
        # We handle replay buffer externally in our training loop, 
        # but NM512's Dreamer pulls from 'dataset' iterator in its __call__ during training.
        # customization: We will modify how we call the agent to avoid using its internal dataset loop 
        # or we provide a dummy one and call _train explicitly.
        
        self.dataset = iter([]) # Dummy
        
        # Disable internal training loop by setting train_every to a large number
        # We manually call _train() in our train_step()
        self.config.train_every = 10**9
        self.config.pretrain = 0
        # NM512 uses train_ratio to calculate training frequency: Every(batch_steps / train_ratio)
        # We want frequency period to be huge, so train_ratio must be tiny
        self.config.train_ratio = 1e-10
        
        # Disable internal logging and video prediction which requires dataset access
        self.config.log_every = 10**9
        self.config.video_pred_log = False
        
        # Initialize Dreamer
        self.agent = dreamer.Dreamer(
            self.obs_space,
            self.act_space,
            self.config,
            self.logger,
            self.dataset
        ).to(self.device)
        
        # Monkeypatch internal training triggers to completely disable them
        # This prevents Dreamer from trying to pull from the empty dataset
        self.agent._should_train = lambda step: False
        self.agent._should_pretrain = lambda: False
        
        # Disable compilation for stability, unless explicitly requested and safe
        if self.config.get('compile', False) and hasattr(torch, "compile"):
             print("Warning: torch.compile enabled. This may interfere with accessing internal methods like _train.")
             self.agent = torch.compile(self.agent)
        
        # Training state
        self.state = None
        
    def select_action(self, obs, eval_mode=False):
        """
        Select action for environment interaction.
        Args:
            obs: Observation (C, H, W) numpy array, [0,1] normalized or uint8
            eval_mode: Boolean, True for evaluation (deterministic)
        """
        # Prepare observation dict
        # NM512 expects (batch, C, H, W) ? No, usually (batch, H, W, C) or flattened keys?
        # Let's check models.py in NM512... it uses tools.SymLog (usually).
        # And it expects 'image' key.
        # Input should be Tensor on device.
        
        with torch.no_grad():
            # Check if batch dimension exists
            if obs.ndim == 3:
                obs = obs[None, ...] # Add batch dim
            
            # Convert to Tensor
            # Handle float [0, 1] vs uint8 [0, 255]
            if obs.dtype == np.uint8:
                 obs_uint8 = obs
            else:
                 # Assume float [0, 1]
                 obs_uint8 = (obs * 255).astype(np.uint8)
            
            data = {'image': torch.tensor(obs_uint8, device=self.device)}
            
            # Helper keys required by NM512 preprocess requirements
            # is_first: True if state is None (first step of episode)
            data['is_first'] = torch.tensor([self.state is None], device=self.device)
            # is_terminal: Always False for action selection (we don't select action on terminal obs usually, 
            # and if we did, it's start of next ep if reset? No, terminal obs is end.)
            # For rollout, we assume we are continuing or starting.
            data['is_terminal'] = torch.tensor([False], device=self.device)
            
            # Action selection
            # Dreamer.__call__ logic:
            # policy_output, state = self._policy(obs, state, training)
            
            training = not eval_mode
            policy_output, self.state = self.agent(data, reset=torch.tensor([False], device=self.device), state=self.state, training=training)
            
            # Extract action
            # NM512 returns one-hot if discrete
            action = policy_output['action'].cpu().numpy()
            
            # Return index
            return np.argmax(action[0])

    def train_step(self, batch):
        """
        Train using a batch from our ReplayBuffer.
        We need to adapt our batch (dict of tensors) to what NM512 expects.
        """
        # NM512 expects a dictionary of sequences: {key: (batch, time, ...)}
        # Our buffer returns: {'obs': (B, T, ...), 'action': (B, T), 'reward': (B, T), ...}
        
        data = {}
        # batch['obs'] is FloatTensor [0, 255] (from ReplayBuffer which converted uint8->Float)
        # We convert back to uint8 tensor for Dreamer (which expects uint8 image usually)
        data['image'] = batch['obs'].to(dtype=torch.uint8, device=self.device)
        data['action'] = torch.nn.functional.one_hot(batch['action'].long(), self.config.action_dim).to(self.device)
        data['reward'] = batch['reward'].to(self.device)
        data['is_first'] = torch.zeros_like(batch['done'], device=self.device) # We assume continuous segments for now
        data['is_terminal'] = batch['done'].to(self.device)
        data['is_last'] = batch['done'].to(self.device)
        
        # _train returns nothing, but updates self.agent._metrics
        self.agent._train(data)
        
        # Retrieve latest metrics from internal storage
        # metrics is a dict of lists
        metrics = {k: v[-1] for k, v in self.agent._metrics.items()}
        
        # Log to internal logger (optional, or we use our own)
        # self.logger.step += ...
        
        return metrics

    def reset_state(self):
        self.state = None

    def save(self, path):
        torch.save(self.agent.state_dict(), path)

    def load(self, path):
        checkpoint = torch.load(path)
        # Handle full checkpoint vs state dict
        if 'agent_state_dict' in checkpoint:
            self.agent.load_state_dict(checkpoint['agent_state_dict'])
        else:
            self.agent.load_state_dict(checkpoint)

    def _recursive_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._recursive_update(d[k], v)
            else:
                d[k] = v

    def _ensure_attribute_dict(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = AttributeDict(v)
                self._ensure_attribute_dict(d[k])
