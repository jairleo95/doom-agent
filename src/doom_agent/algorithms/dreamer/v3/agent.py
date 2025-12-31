"""
Dreamer V3 Agent Adapter for NM512 implementation.

Wraps the NM512/dreamerv3-torch implementation to fit our interface.
"""

import sys
import pathlib
import torch
import numpy as np
import ruamel.yaml as yaml

import os

# Add NM512 repo directory to path so we can import its modules directly
# Use .resolve() to ensure we have an absolute path
nm512_path = pathlib.Path(__file__).parent.resolve() / "nm512_dreamer"

if str(nm512_path) not in sys.path:
    sys.path.insert(0, str(nm512_path))

try:
    import tools
    import dreamer
    import exploration as expl
except ImportError:
    print("Error: Could not import NM512 Dreamer implementation.")
    print(f"Make sure {nm512_path} exists and contains tools.py, dreamer.py, etc.")
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

        # Map action_dim to num_actions for NM512 compatibility
        if 'action_dim' in config:
            self.config.num_actions = config['action_dim']
        
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
        
    def select_action(self, obs, eval_mode=False, deterministic=None, is_first=None):
        """
        Select action for environment interaction.
        Args:
            obs: Observation (H, W, 1) or (N, H, W, 1) numpy array, uint8
            eval_mode: Boolean, True for evaluation (deterministic)
            deterministic: Alias for eval_mode (used by some callbacks)
            is_first: Boolean or boolean array/list, True if starting a new episode
        """
        if deterministic is not None:
            eval_mode = deterministic
            
        with torch.no_grad():
            # Ensure obs is (N, H, W, 1)
            if obs.ndim == 3:
                obs = obs[None, ...]
                n_envs = 1
            else:
                n_envs = obs.shape[0]
            
            # Check if batch size changed and reset state if so
            if self.state is not None:
                # self.state is (latent, action)
                # latent is a dict of tensors
                latent_state = self.state[0]
                if isinstance(latent_state, dict):
                    expected_batch = list(latent_state.values())[0].shape[0]
                else:
                    expected_batch = latent_state.shape[0]
                    
                if expected_batch != n_envs:
                    self.reset_state()
            
            # Handle is_first
            if is_first is None:
                # If not provided, we infer from self.state being None (only works for n_envs=1)
                is_first = [self.state is None] * n_envs
            elif isinstance(is_first, bool):
                is_first = [is_first] * n_envs
            
            data = {'image': torch.as_tensor(obs, device=self.device)}
            data['is_first'] = torch.as_tensor(is_first, device=self.device)
            # is_terminal/is_last not used in policy selection by Dreamer NM512
            data['is_terminal'] = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
            
            training = not eval_mode
            # NM512 Dreamer call: __call__(obs, reset, state=None, training=True)
            # Note: NM512 Dreamer reset arg is the same as is_first in this context
            policy_output, self.state = self.agent(data, reset=data['is_first'], state=self.state, training=training)
            
            # Extract actions
            action = policy_output['action'].cpu().numpy()
            
            # Return list of indices if n_envs > 1, else single index
            if n_envs > 1:
                return np.argmax(action, axis=-1)
            else:
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
        data['is_first'] = batch['is_first'].to(self.device)
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
        checkpoint = torch.load(path, map_location=self.device)
        # Handle full checkpoint vs state dict
        if 'agent_state_dict' in checkpoint:
            state_dict = checkpoint['agent_state_dict']
        else:
            state_dict = checkpoint

        # Strip torch.compile "_orig_mod." prefixes from checkpoint for a "clean" version
        checkpoint_clean = {}
        for k, v in state_dict.items():
            clean_name = k.replace("_orig_mod.", "")
            checkpoint_clean[clean_name] = v
        
        # Get the model's current keys and create a mapping from their clean versions to actual names
        model_keys = self.agent.state_dict().keys()
        clean_to_model = {k.replace("_orig_mod.", ""): k for k in model_keys}
        
        # Build the final state_dict by aligning clean checkpoint keys with model keys
        final_state_dict = {}
        for clean_name, value in checkpoint_clean.items():
            if clean_name in clean_to_model:
                actual_name = clean_to_model[clean_name]
                final_state_dict[actual_name] = value
        
        # Load with mismatched keys ignored if necessary (though our mapping should be exact)
        self.agent.load_state_dict(final_state_dict, strict=False)

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
