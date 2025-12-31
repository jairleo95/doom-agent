
import os
import sys
import numpy as np
import imageio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[6] / "src"))
# Local import fix
sys.path.append(str(Path(__file__).resolve().parent))

from doom_agent.algorithms.dreamer.v3.doom_envs import DoomDreamerEnv, defend_actions

def test_video_generation():
    print("Testing Video Generation Logic...")
    
    # Create a dummy observation (64, 64, 1) uint8
    obs = np.random.randint(0, 256, (64, 64, 1), dtype=np.uint8)
    
    # Simulate the bug in callbacks.py:58
    try:
        frame_buggy = (obs[0] * 255).astype(np.uint8)
        print(f"Buggy frame shape: {frame_buggy.shape} (Expected (64, 64) or (64, 64, 3))")
    except Exception as e:
        print(f"Buggy code failed as expected: {e}")

    # Fix logic
    frame_fixed = obs.squeeze(-1)
    print(f"Fixed frame shape: {frame_fixed.shape}")
    
    # Test imageio saving
    test_file = "test_output.gif"
    frames = [frame_fixed for _ in range(10)]
    imageio.mimsave(test_file, frames, fps=10)
    
    if os.path.exists(test_file):
        print(f"Success: {test_file} generated. Size: {os.path.getsize(test_file)} bytes")
        os.remove(test_file)
    else:
        print("Failure: Video not generated.")

if __name__ == "__main__":
    test_video_generation()
