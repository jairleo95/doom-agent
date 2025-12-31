
import unittest
import numpy as np
import torch
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
project_root = Path(__file__).resolve().parents[5]
sys.path.append(str(project_root / "src"))
# Local import fix
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from doom_agent.algorithms.dreamer.v3.replay_buffer import ReplayBuffer
from doom_agent.algorithms.dreamer.v3.callbacks import MetricsCallback, ImaginationVideoCallback
from doom_agent.algorithms.dreamer.v3.train import flip_actions

class TestAdvancedFeatures(unittest.TestCase):
    
    def test_replay_buffer_flip(self):
        """Test that horizontal flipping works and doesn't have stride issues."""
        seq_len = 5
        buffer = ReplayBuffer(capacity=100, obs_shape=(4, 4, 3), sequence_length=seq_len)
        
        # Add some data: a 4x4 image with a pattern
        # Left side is 1s, right side is 0s
        obs = np.zeros((4, 4, 3), dtype=np.uint8)
        obs[:, :2, :] = 255 
        
        for _ in range(seq_len + 1):
            buffer.add(obs, 0, 0.0, 0.0, False)
            
        # Sample with flip
        batch = buffer.sample(batch_size=1, horizontal_flip=True)
        
        self.assertIsNotNone(batch)
        obs_batch = batch['obs'] # torch.Tensor
        
        # Check shape (B, T, H, W, C)
        self.assertEqual(obs_batch.shape, (1, seq_len, 4, 4, 3))
        
        # Check if flipped: Left side should now be 0s, right side 255s
        # Original: [255, 255, 0, 0]
        # Flipped: [0, 0, 255, 255]
        flipped_obs = obs_batch[0, 0].numpy()
        self.assertEqual(flipped_obs[0, 0, 0], 0)
        self.assertEqual(flipped_obs[0, 3, 0], 255)
        
        # Regression test for stride issue: torch.as_tensor on a flipped array
        # should not crash if we used .copy() in the implementation
        # (The test above already confirms it didn't crash because obs_batch existed)
        
    def test_flip_actions(self):
        """Test that action remapping for symmetry is correct."""
        # Universal set indices: 3:TL, 4:TR, 5:TL+ATK, 6:TR+ATK, 7:SL, 8:SR, 10:FWD+TL, 11:FWD+TR
        # TL (3) -> TR (4)
        a = torch.tensor([3, 4, 5, 6, 7, 8, 10, 11, 0, 1, 2, 9])
        expected = torch.tensor([4, 3, 6, 5, 8, 7, 11, 10, 0, 1, 2, 9])
        
        flipped = flip_actions(a)
        torch.testing.assert_close(flipped, expected)

    def test_metrics_callback_detailed(self):
        """Test that MetricsCallback logs gameplay-specific metrics."""
        # Need to patch the class where it's imported
        patch_target = 'doom_agent.algorithms.dreamer.v3.callbacks.SummaryWriter'
        with patch(patch_target) as mock_writer_class:
            mock_writer = mock_writer_class.return_value
            callback = MetricsCallback(log_path="test_logs")
            
            info = {'frags': 10, 'health': 80, 'ammo': 50}
            callback.log_episode(1, 100.0, 200, 10.5, step=1000, info=info)
            
            # Check if add_scalar was called for gameplay metrics
            calls = [call[0][0] for call in mock_writer.add_scalar.call_args_list]
            self.assertIn('gameplay/frags', calls)
            self.assertIn('gameplay/health_remaining', calls)
            self.assertIn('gameplay/ammo_consumed', calls)

    def test_imagination_callback(self):
        """Test ImaginationVideoCallback initialization and should_render."""
        agent = MagicMock()
        log_dir = "test_imag_logs"
        callback = ImaginationVideoCallback(agent, log_dir, render_freq=100)
        
        # should_render now just checks if render_freq > 0 (caller manages frequency)
        self.assertTrue(callback.should_render(100))
        self.assertTrue(callback.should_render(50)) 
        self.assertTrue(callback.should_render(200))

if __name__ == '__main__':
    unittest.main()
