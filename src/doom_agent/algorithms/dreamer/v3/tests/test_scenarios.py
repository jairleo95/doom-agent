
"""
Integration Tests for All DreamerV3 Scenarios
Runs a short training loop for each scenario to ensure no runtime errors.
"""

import sys
import unittest
from pathlib import Path
import torch
import shutil
import tempfile

# Add src to path
# Assuming this script is in src/doom_agent/algorithms/dreamer/v3/tests/
sys.path.append(str(Path(__file__).resolve().parents[6] / "src"))
# Local import fix
sys.path.append(str(Path(__file__).resolve().parent))

from doom_agent.algorithms.dreamer.v3.train import main as train_main
from unittest.mock import patch

class TestScenarios(unittest.TestCase):
    
    def setUp(self):
        # Create a temp dir for logs to avoid clutter
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def run_scenario(self, scenario_name):
        print(f"\nTesting Scenario: {scenario_name}")
        # Run 1 step of training for 1 episode (minimal dry run)
        # We mock sys.argv
        args = [
            "train.py",
            "--scenario", scenario_name,
            "--batch-size", "2", # Small batch
            "--batch-length", "4", # Short seq
            "--buffer-capacity", "100",
            "--prefill-steps", "10", # Minimal prefill
            "--train-every", "5",
            "--video-freq", "10", # Trigger video recording during test
            "--device", "cpu" # Force CPU for testing
        ]
        
        with patch.object(sys, 'argv', args):
            # Hack: Patch argparse in train.py? No, main() parses sys.argv.
            # But main() catches sys.exit? No.
            # We need to ensure main() doesn't run forever.
            # train.py loops based on curriculum.
            # We can't easily modify the curriculum loop from outside.
            # BUT we can modify the CURRICULUM object itself before calling main!
            
            from doom_agent.algorithms.dreamer.v3 import train
            from doom_agent.algorithms.dreamer.v3.curriculum import Stage, Curriculum
            
            # Override curriculum to be extremely short
            mock_curriculum = Curriculum(
                name=f"test_{scenario_name}",
                scenario=f"{scenario_name}.cfg",
                stages=[
                    Stage(
                        name="test_stage",
                        timesteps=20, # Run for 20 steps only
                        doom_skill=1,
                        living_reward=0.0
                    )
                ]
            )
            
            # Patch the constants in train module
            if scenario_name == 'deathmatch':
                with patch.object(train, 'DEATHMATCH_CURRICULUM', mock_curriculum):
                    train.main()
            elif scenario_name == 'deadly_corridor':
                with patch.object(train, 'DEADLY_CORRIDOR_CURRICULUM', mock_curriculum):
                    train.main()
            elif scenario_name == 'defend_the_center':
                with patch.object(train, 'DEFEND_CENTER_CURRICULUM', mock_curriculum):
                    train.main()
                    
    def test_deathmatch(self):
        self.run_scenario("deathmatch")
        
    def test_deadly_corridor(self):
        self.run_scenario("deadly_corridor")
        
    def test_defend_the_center(self):
        self.run_scenario("defend_the_center")

if __name__ == "__main__":
    unittest.main()
