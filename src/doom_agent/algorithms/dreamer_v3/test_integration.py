import sys
import shutil
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[4] # Adjust levels to reach src/
sys.path.append(str(project_root / "src"))

from doom_agent.algorithms.dreamer_v3.agent import DreamerV3Agent
import numpy as np
import torch

def test_integration():
    print("Testing DreamerV3 Integration...")
    
    # Mock config
    config = {
        'batch_size': 2,
        'batch_length': 10,
        'obs_shape': (64, 64, 1),
        'action_dim': 4,
        'train_every': 5,
        'device': 'cpu',
        'compile': False,
        'precision': 32,
        'logdir': './test_run',
        'traindir': './test_run/train',
        'evaldir': './test_run/eval',
        'dyn_hidden': 64,
        'dyn_stoch': 4, 
        'dyn_discrete': 4,
        'units': 64,
        'encoder': {'mlp_keys': '$^', 'cnn_keys': 'image', 'cnn_depth': 16, 'mlp_units': 64},
        'decoder': {'mlp_keys': '$^', 'cnn_keys': 'image', 'cnn_depth': 16, 'mlp_units': 64},
        
        # Add required params usually in config
        'action_repeat': 1,
        'num_actions': 4,
        'envs': 1,
        'task': 'dummy_task'
    }
    
    run_dir = Path('./test_run')
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir()
    
    try:
        # Initialize agent
        print("Initializing agent...")
        agent = DreamerV3Agent(config, run_dir=str(run_dir))
        print("Agent initialized successfully.")
        
        # Test select_action
        print("Testing select_action...")
        obs = np.zeros((64, 64, 1), dtype=np.uint8) # (H, W, C) uint8 [0,255]
        action = agent.select_action(obs)
        print(f"Action selected: {action}")
        
        # Test train_step
        print("Testing train_step...")
        batch = {
            'obs': torch.zeros((2, 10, 64, 64, 1)), # (B, T, H, W, C)
            'action': torch.randint(0, 4, (2, 10)),
            'reward': torch.zeros((2, 10)),
            'done': torch.zeros((2, 10))
        }
        
        metrics = agent.train_step(batch)
        print(f"Training metrics: {list(metrics.keys())}")
        
        print("\nIntegration test PASSED!")
        
    except Exception as e:
        print(f"\nIntegration test FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if run_dir.exists():
            shutil.rmtree(run_dir)

if __name__ == "__main__":
    test_integration()
