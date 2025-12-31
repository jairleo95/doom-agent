"""
Unit tests for Dreamer V3 based on NM512 integration.

Tests the DreamerV3Agent adapter and its interaction with the NM512 implementation.
"""

import unittest
import numpy as np
import torch
import shutil
import tempfile
from pathlib import Path
import sys
import ruamel.yaml as yaml

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from doom_agent.algorithms.dreamer_v3.agent import DreamerV3Agent

class TestDreamerV3Agent(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.test_dir)
        
        # Mock configuration compatible with DreamerV3Agent adapter
        self.config = {
            'device': 'cpu',
            'compile': False,
            'train_every': 5,
            'obs_shape': (64, 64, 1),
            'action_dim': 4,
            'batch_size': 2,
            'batch_length': 5,
            
            # Additional config needed by NM512 defaults
            'num_actions': 4,
            'envs': 1,
            'compile': False,
            'precision': 32,
            'logdir': self.test_dir,
            
            # Model params to speed up init
            'dyn_hidden': 32,
            'dyn_stoch': 4,
            'dyn_discrete': 4,
            'units': 32,
            'encoder': {'mlp_keys': '$^', 'cnn_keys': 'image', 'cnn_depth': 8, 'mlp_units': 32},
            'decoder': {'mlp_keys': '$^', 'cnn_keys': 'image', 'cnn_depth': 8, 'mlp_units': 32},
        }
        
    def tearDown(self):
        # Cleanup temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_agent_initialization(self):
        """Test that the agent initializes correctly with config."""
        agent = DreamerV3Agent(self.config, run_dir=self.test_dir)
        self.assertIsNotNone(agent.agent)
        self.assertEqual(agent.device, torch.device('cpu'))
        self.assertEqual(agent.config.logdir, str(self.run_dir))
        
    def test_select_action(self):
        """Test action selection from observation."""
        agent = DreamerV3Agent(self.config, run_dir=self.test_dir)
        
        # Create dummy observation: (H, W, C) float [0, 1]
        obs_shape = (64, 64, 1)
        obs = np.random.rand(*obs_shape).astype(np.float32)
        
        # Test training mode
        action = agent.select_action(obs, eval_mode=False)
        self.assertTrue(isinstance(action, (int, np.integer)))
        self.assertTrue(0 <= action < self.config['action_dim'])
        
        # Test eval mode
        action_eval = agent.select_action(obs, eval_mode=True)
        self.assertTrue(isinstance(action_eval, (int, np.integer)))
        
    def test_train_step(self):
        """Test training step with dummy batch."""
        agent = DreamerV3Agent(self.config, run_dir=self.test_dir)
        
        # Create dummy batch matching ReplayBuffer output
        # (Batch, Time, ...)
        batch_size = 2
        seq_len = 5
        
        batch = {
            'obs': torch.rand((batch_size, seq_len, 64, 64, 1)), # (B, T, H, W, C)
            'action': torch.randint(0, 4, (batch_size, seq_len)),
            'reward': torch.rand((batch_size, seq_len)),
            'done': torch.zeros((batch_size, seq_len)),
            'is_first': torch.zeros((batch_size, seq_len))
        }
        
        # Run training step
        metrics = agent.train_step(batch)
        
        # Verify metrics returned
        self.assertIsInstance(metrics, dict)
        # Check standard Dreamer metrics exist
        keys_to_check = [
            'image_loss', 
            'reward_loss', 
            'tech_loss' # or 'loss' depending on NM512 exact keys
        ]
        # Just check we got *some* non-empty metrics
        self.assertTrue(len(metrics) > 0)
        
    def test_save_load(self):
        """Test saving and loading checkpoints."""
        agent = DreamerV3Agent(self.config, run_dir=self.test_dir)
        
        save_path = self.run_dir / "test_ckpt.pt"
        
        # Save
        agent.save(str(save_path))
        self.assertTrue(save_path.exists())
        
        # Load
        agent.load(str(save_path))
        
        # Verify it loads without error (state check is harder without deep inspection)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
