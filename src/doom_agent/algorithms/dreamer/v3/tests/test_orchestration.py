import unittest
import shutil
import tempfile
from pathlib import Path
import sys
from omegaconf import OmegaConf

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[6]
sys.path.append(str(project_root / "src"))

from doom_agent.algorithms.dreamer.v3.experiment import ExperimentManager
from doom_agent.algorithms.dreamer.v3.trainer import DreamerV3Trainer
from doom_agent.algorithms.dreamer.v3.curriculum import Curriculum, Stage

class TestOrchestration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.run_dir = Path(self.test_dir)
        
        # Mock Hydra config
        self.cfg = OmegaConf.create({
            'scenario': {
                'name': 'test_scenario',
                'scenario_name': 'test_map',
                'curriculum': {
                    'stages': [{'name': 'stage1', 'timesteps': 100, 'doom_skill': 3, 'living_reward': 0.0}]
                }
            },
            'agent': {
                'obs_shape': [64, 64, 3],
                'batch_size': 2,
                'batch_length': 5,
                'n_envs': 1,
                'train_every': 5,
                'train_steps': 1,
                'prefill_steps': 10,
                'buffer_capacity': 1000
            },
            'wandb': {
                'enabled': False,
                'save_artifacts': False,
                'project': 'test_project',
                'entity': None,
                'group': 'test_group',
                'name': None,
                'mode': 'disabled'
            },
            'compile': False,
            'device': 'cpu',
            'visualize': False,
            'resume': None,
            'start_stage': 0,
            'video_freq': 0
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_experiment_manager_paths(self):
        """Test that ExperimentManager creates correct directory structure."""
        # We need to mock Path(__file__) in ExperimentManager or adjust its base_dir
        # Since it uses Path(__file__).resolve().parent, it will use the real package dir.
        # For testing, we might want to pass base_dir or just verify relative structures.
        exp = ExperimentManager(self.cfg)
        
        self.assertTrue(exp.log_dir.exists())
        self.assertTrue(exp.ckpt_dir.exists())
        self.assertTrue(exp.video_dir.exists())
        self.assertIn("test_scenario", str(exp.log_dir))

    def test_experiment_manager_config_persistence(self):
        """Test saving config and updating manifest."""
        exp = ExperimentManager(self.cfg)
        curriculum = Curriculum(name="test", scenario="test.cfg", stages=[Stage(name="s1", timesteps=10, doom_skill=3, living_reward=0.0)])
        
        exp.save_config(curriculum)
        config_file = exp.log_dir / "config.json"
        self.assertTrue(config_file.exists())
        
        exp.update_manifest("test_curriculum")
        manifest_file = exp.log_dir.parent.parent / "experiments_manifest.csv"
        self.assertTrue(manifest_file.exists())

    def test_trainer_initialization(self):
        """Test Trainer initializes components correctly."""
        exp = ExperimentManager(self.cfg)
        curriculum = Curriculum(name="test", scenario="test.cfg", stages=[Stage(name="s1", timesteps=10, doom_skill=3, living_reward=0.0)])
        actions = [0, 1, 2]
        
        trainer = DreamerV3Trainer(self.cfg, exp, curriculum, actions)
        self.assertEqual(len(trainer.actions), 3)
        self.assertEqual(trainer.global_step, 0)
        self.assertIsNotNone(trainer.agent)

if __name__ == '__main__':
    unittest.main()
