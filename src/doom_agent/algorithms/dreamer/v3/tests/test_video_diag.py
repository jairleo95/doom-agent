
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
    print("Testing Video Generation Logic (RGB)...")
    
    # Create a dummy observation (64, 64, 3) uint8
    obs = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    # In RGB, the frame matches the observation
    frame_fixed = obs
    print(f"Fixed frame shape: {frame_fixed.shape}")
    
    # Test imageio saving
    test_file = "test_output.gif"
    frames = [frame_fixed for _ in range(10)]
    # Use duration instead of fps for newer imageio/pillow
    imageio.mimsave(test_file, frames, duration=100) # 100ms = 10fps
    
    if os.path.exists(test_file):
        print(f"Success: {test_file} generated. Size: {os.path.getsize(test_file)} bytes")
        os.remove(test_file)
    else:
        print("Failure: Video not generated.")

if __name__ == "__main__":
    test_video_generation()
