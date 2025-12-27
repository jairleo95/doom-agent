"""
Experience replay buffer for Dreamer V3.
Optimized with NumPy arrays for O(1) sampling.
"""

import numpy as np
import torch

class ReplayBuffer:
    """Experience replay buffer for Dreamer V3 optimized for high throughput."""
    
    def __init__(self, capacity=1_000_000, sequence_length=50, obs_shape=(64, 64, 1)):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.obs_shape = obs_shape
        
        # Pre-allocate numpy arrays for memory efficiency
        self.obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.is_firsts = np.zeros(capacity, dtype=np.float32)
        
        self.idx = 0
        self.size = 0
        self.full = False
        
    def add(self, observation, action, reward, done, is_first):
        """Add a single transition."""
        self.obs[self.idx] = observation
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.is_firsts[self.idx] = is_first
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or (self.idx == 0)
        self.size = self.capacity if self.full else self.idx
    
    def sample(self, batch_size):
        """Sample sequences of experiences using vectorized lookups."""
        if self.size <= self.sequence_length:
            return None
            
        # Sample starting indices
        max_idx = self.size - self.sequence_length
        start_indices = np.random.randint(0, max_idx, size=batch_size)
        
        # Create sequence offsets: [0, 1, ..., seq_len-1]
        offsets = np.arange(self.sequence_length)
        
        # Compute 2D indices: (batch_size, sequence_length)
        indices = start_indices[:, None] + offsets[None, :]
        
        # Vectorized lookup
        batch = {
            'obs': torch.as_tensor(self.obs[indices], dtype=torch.uint8),
            'action': torch.as_tensor(self.actions[indices], dtype=torch.long),
            'reward': torch.as_tensor(self.rewards[indices], dtype=torch.float32),
            'done': torch.as_tensor(self.dones[indices], dtype=torch.float32),
            'is_first': torch.as_tensor(self.is_firsts[indices], dtype=torch.float32)
        }
        
        return batch
    
    def __len__(self):
        return self.size
