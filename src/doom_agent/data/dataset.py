
import os
import torch
import numpy as np
from torch.utils.data import IterableDataset
from pathlib import Path
import random

class OfflineDoomDataset(IterableDataset):
    """
    Streams offline data from collected .npz files for Behavior Cloning / Offline RL.
    Yields sequences of length `seq_length`.
    """
    def __init__(self, data_dir, seq_length=50, action_dim=7):
        """
        Args:
            data_dir: Directory containing .npz files
            seq_length: Length of sequences to yield
            action_dim: Number of discrete action indices (default 7 for Universal)
        """
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.npz")))
        self.seq_length = seq_length
        self.action_dim = action_dim
        
        if not self.files:
            print(f"⚠️ Warning: No .npz files found in {data_dir}")

    def __iter__(self):
        """Yields single sequences (T, ...)"""
        # Shuffle files each epoch
        files = list(self.files)
        random.shuffle(files)
        
        for file_path in files:
            try:
                with np.load(file_path) as data:
                    obs = data['obs']          # (N, H, W, C)
                    actions = data['actions']  # (N, 7) or (N,)
                    rewards = data['rewards']  # (N,)
                    dones = data['dones']      # (N,)
                    
                n_steps = len(obs)
                
                # We need at least seq_length steps
                if n_steps < self.seq_length:
                    continue
                    
                # Slice into non-overlapping sequences
                for i in range(0, n_steps - self.seq_length, self.seq_length):
                    idx_end = i + self.seq_length
                    
                    # 1. Prepare Observations
                    seq_obs = obs[i:idx_end]
                    
                    # 2. Prepare Actions
                    seq_actions_vec = actions[i:idx_end]
                    
                    # Convert Vector to Index (Argmax) if inputs are vectors
                    if seq_actions_vec.ndim > 1:
                        # (T, 7) -> (T,)
                        seq_actions = np.argmax(seq_actions_vec, axis=-1)
                    else:
                        seq_actions = seq_actions_vec
                        
                    # 3. Prepare Rewards/Dones
                    seq_rewards = rewards[i:idx_end]
                    seq_dones = dones[i:idx_end]
                    
                    # 4. Synthesize 'is_first'
                    seq_is_first = np.zeros_like(seq_dones)
                    if i == 0:
                        seq_is_first[0] = 1.0
                    
                    # Yield dictionary identical to what DreamerV3Agent expects from batch
                    # Note: DreamerV3Agent.train_step() converts 'obs' -> 'image'.
                    # ReplayBuffer yields 'obs'. We yield 'obs'.
                    yield {
                        'obs': seq_obs,         # (T, 64, 64, 3)
                        'action': seq_actions,  # (T,)
                        'reward': seq_rewards,  # (T,)
                        'done': seq_dones,      # (T,)
                        'is_first': seq_is_first # (T,)
                    }

            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue

    def __len__(self):
        # Approximate length (files * 20 sequences?)
        return len(self.files) * 20
