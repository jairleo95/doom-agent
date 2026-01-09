
import unittest
import numpy as np
import torch
import shutil
import tempfile
from pathlib import Path
import sys
import os

# Add src to sys.path
root_dir = Path(__file__).resolve().parents[6]
sys.path.append(str(root_dir / "src"))

from doom_agent.data.dataset import OfflineDoomDataset

class TestBehaviorCloning(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.test_dir / "data"
        self.data_dir.mkdir()
        
        # Create a dummy NPZ file
        self.create_dummy_npz(self.data_dir / "chunk_0000.npz", n_steps=120)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_dummy_npz(self, path, n_steps=100):
        obs = np.random.randint(0, 255, (n_steps, 64, 64, 3), dtype=np.uint8)
        actions = np.random.randint(0, 2, (n_steps, 7), dtype=np.int8)
        rewards = np.random.rand(n_steps).astype(np.float32)
        dones = np.zeros(n_steps, dtype=bool)
        dones[-1] = True
        
        np.savez_compressed(path, obs=obs, actions=actions, rewards=rewards, dones=dones)

    def test_dataset_loading(self):
        """Test that OfflineDoomDataset loads and slices data correctly."""
        subdir = self.data_dir / "loading"
        subdir.mkdir()
        self.create_dummy_npz(subdir / "chunk_0000.npz", n_steps=120)

        seq_len = 50
        dataset = OfflineDoomDataset(subdir, seq_length=seq_len)
        
        # We created 120 steps, so it should yield 2 sequences of 50
        count = 0
        for batch in dataset:
            self.assertEqual(batch['obs'].shape, (seq_len, 64, 64, 3))
            self.assertEqual(batch['action'].shape, (seq_len,))
            self.assertEqual(batch['reward'].shape, (seq_len,))
            self.assertEqual(batch['done'].shape, (seq_len,))
            self.assertIn('is_first', batch)
            count += 1
            
        self.assertEqual(count, 2)

    def test_action_conversion(self):
        """Test that action vectors are correctly converted to indices via argmax."""
        subdir = self.data_dir / "action"
        subdir.mkdir()
        
        # Create a chunk with known action vectors
        n_steps = 60
        actions = np.zeros((n_steps, 7), dtype=np.int8)
        actions[:, 6] = 1 # All ATTACK
        
        path = subdir / "chunk_action_test.npz"
        obs = np.zeros((n_steps, 64, 64, 3), dtype=np.uint8)
        np.savez_compressed(path, obs=obs, actions=actions, rewards=np.zeros(n_steps), dones=np.zeros(n_steps))
        
        dataset = OfflineDoomDataset(subdir, seq_length=50)
        
        found = False
        for batch in dataset:
            found = True
            # np.argmax([0,0,0,0,0,0,1]) should be 6
            np.testing.assert_array_equal(batch['action'], np.full(50, 6))
        self.assertTrue(found)

if __name__ == '__main__':
    unittest.main()
