
"""
Integration Tests for All DreamerV3 Scenarios
Runs a short training loop for each scenario to ensure no runtime errors.
Uses subprocess to avoid global state pollution and test the script entry point directly.
"""

import sys
import unittest
import subprocess
import shutil
import tempfile
import os
from pathlib import Path

class TestScenarios(unittest.TestCase):
    
    def setUp(self):
        # Create a temp dir for logs to avoid clutter
        self.test_dir = tempfile.mkdtemp()
        
        # Resolve python executable and train script
        self.python_exe = sys.executable
        # Assuming we are in src/doom_agent/.../tests/
        project_root = Path(__file__).resolve().parents[6]
        self.train_script = project_root / "src/doom_agent/algorithms/dreamer/v3/train.py"
        self.src_dir = project_root / "src"
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def run_scenario(self, scenario_name):
        print(f"\nTesting Scenario: {scenario_name}")
        
        # Improvements: Run for minimal steps to check crash loop
        # Hydra syntax for overrides
        cmd = [
            self.python_exe, str(self.train_script),
            f"scenario={scenario_name}",
            "agent.batch_size=2",
            "agent.batch_length=4",
            "agent.prefill_steps=10",
            "agent.train_every=5",
            "agent.n_envs=1",
            "device=cpu",
            f"hydra.run.dir={self.test_dir}/{scenario_name}",
            # Override curriculum to be extremely short
            # Override curriculum to be extremely short
            "scenario.curriculum.stages.0.timesteps=5"
        ]
        
        # Add stage overrides based on scenario
        if scenario_name == "defend_the_center":
            cmd.append("scenario.curriculum.stages.1.timesteps=5")
        elif scenario_name == "deadly_corridor":
             cmd.append("scenario.curriculum.stages.1.timesteps=5")
             cmd.append("scenario.curriculum.stages.2.timesteps=5")
        elif scenario_name == "deathmatch":
             # deathmatch.yaml is single stage
             pass
        
        # Ensure PYTHONPATH is set so that imports work
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.src_dir) + ":" + env.get("PYTHONPATH", "")
        
        # Run process with timeout
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                timeout=180  # 180 seconds timeout
            )
            
            if result.returncode != 0:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                
            self.assertEqual(result.returncode, 0, f"Training failed for {scenario_name}")
            
        except subprocess.TimeoutExpired as e:
            self.fail(f"Training timed out for {scenario_name} after 180s. STDOUT: {e.stdout} STDERR: {e.stderr}")
        
    def test_deathmatch(self):
        self.run_scenario("deathmatch")
        
    def test_deadly_corridor(self):
        self.run_scenario("deadly_corridor")
        
    def test_defend_the_center(self):
        self.run_scenario("defend_the_center")

if __name__ == "__main__":
    unittest.main()
