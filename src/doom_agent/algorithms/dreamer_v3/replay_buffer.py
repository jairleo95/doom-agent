"""
Experience replay buffer for Dreamer V3.

Stores transitions and samples sequences for training the world model.
"""

from collections import deque
import numpy as np
import torch
import os


class ReplayBuffer:
    """Experience replay buffer for Dreamer V3."""
    
    def __init__(self, capacity=1_000_000, sequence_length=50):
        """
        Args:
            capacity: Maximum number of transitions to store
            sequence_length: Length of sequences to sample
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        
    def add(self, observation, action, reward, done, is_first):
        """Add a single transition."""
        self.buffer.append({
            'obs': observation,
            'action': action,
            'reward': reward,
            'done': done,
            'is_first': is_first
        })
    
    def sample(self, batch_size):
        """Sample sequences of experiences."""
        if len(self.buffer) < self.sequence_length:
            return None
            
        sequences = []
        for _ in range(batch_size):
            # Sample random starting point
            start_idx = np.random.randint(0, len(self.buffer) - self.sequence_length)
            sequence = {
                'obs': [],
                'action': [],
                'reward': [],
                'done': [],
                'is_first': []
            }
            
            for i in range(self.sequence_length):
                transition = self.buffer[start_idx + i]
                sequence['obs'].append(transition['obs'])
                sequence['action'].append(transition['action'])
                sequence['reward'].append(transition['reward'])
                sequence['done'].append(transition['done'])
                sequence['is_first'].append(transition['is_first'])
            
            sequences.append(sequence)
        
        # Convert to tensors
        batch = {
            'obs': torch.FloatTensor(np.array([s['obs'] for s in sequences])),
            'action': torch.LongTensor(np.array([s['action'] for s in sequences])),
            'reward': torch.FloatTensor(np.array([s['reward'] for s in sequences])),
            'done': torch.FloatTensor(np.array([s['done'] for s in sequences])),
            'is_first': torch.FloatTensor(np.array([s['is_first'] for s in sequences]))
        }
        
        return batch
    
    def __len__(self):
        return len(self.buffer)

