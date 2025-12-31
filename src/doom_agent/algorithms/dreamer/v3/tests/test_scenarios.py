import sys
import unittest
from pathlib import Path
import torch
import shutil
import tempfile
from omegaconf import OmegaConf

# Add src to path
project_root = Path(__file__).resolve().parents[6]
sys.path.append(str(project_root / "src"))

from doom_agent.algorithms.dreamer.v3.experiment import ExperimentManager
from doom_agent.algorithms.dreamer.v3.trainer import DreamerV3Trainer
from doom_agent.algorithms.dreamer.v3.curriculum import Curriculum, Stage
from doom_agent.algorithms.dreamer.v3.doom_envs import universal_actions, deadly_corridor_actions, defend_actions

class TestScenarios(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def run_scenario(self, scenario_name, actions):
        print(f"\nTesting Scenario: {scenario_name}")
        
        # Minimal Hydra-like config
        cfg = OmegaConf.create({
            'scenario': {
                'name': scenario_name,
                'scenario_name': scenario_name,
                'curriculum': {
                    'stages': [{'name': 'test_stage', 'timesteps': 4, 'doom_skill': 1, 'living_reward': 0.0}]
                }
            },
            'agent': {
                'obs_shape': [64, 64, 3],
                'batch_size': 2,
                'batch_length': 2,
                'n_envs': 1,
                'train_every': 2,
                'train_steps': 1,
                'prefill_steps': 2,
                'buffer_capacity': 100
            },
            'compile': False,
            'wandb': {'enabled': False, 'save_artifacts': False},
            'device': 'cpu',
            'visualize': False,
            'resume': None,
            'start_stage': 0,
            'video_freq': 0
        })
        
        exp = ExperimentManager(cfg)
        # Override exp.base_dir or similar if needed, but tempdir is better
        exp.log_dir = Path(self.test_dir) / "logs"
        exp.ckpt_dir = Path(self.test_dir) / "checkpoints"
        exp.video_dir = exp.ckpt_dir / "videos"
        exp._setup_directories()

        curriculum = Curriculum(
            name=f"test_{scenario_name}",
            scenario=f"{scenario_name}.cfg",
            stages=[Stage(name="test", timesteps=4, doom_skill=1, living_reward=0.0)]
        )
        
        trainer = DreamerV3Trainer(cfg, exp, curriculum, actions)
        # Run a very small training loop
        trainer.run()
        print(f"Scenario {scenario_name} PASSED.")

    def test_deathmatch(self):
        self.run_scenario("deathmatch", universal_actions())
        
    def test_deadly_corridor(self):
        self.run_scenario("deadly_corridor", deadly_corridor_actions())
        
    def test_defend_the_center(self):
        self.run_scenario("defend_the_center", defend_actions())

if __name__ == "__main__":
    unittest.main()
