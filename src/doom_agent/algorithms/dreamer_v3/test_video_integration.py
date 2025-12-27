
import os
import sys
import numpy as np
import imageio
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[4] / "src"))
# Local import fix
sys.path.append(str(Path(__file__).resolve().parent))

from doom_agent.algorithms.dreamer_v3.callbacks import VideoRecorderCallback

def test_video_callback():
    print("Testing VideoRecorderCallback Integration...")
    
    # Mock Env
    eval_env = MagicMock()
    # Return a dummy observation (64, 64, 1) uint8
    dummy_obs = np.random.randint(0, 256, (64, 64, 1), dtype=np.uint8)
    eval_env.reset.return_value = dummy_obs
    eval_env.step.side_effect = [
        (dummy_obs, 0.0, False),
        (dummy_obs, 0.0, False),
        (dummy_obs, 0.0, True),
    ]
    
    # Mock Agent
    agent = MagicMock()
    agent.select_action.return_value = 0
    
    # Callback
    save_path = "test_videos"
    callback = VideoRecorderCallback(
        eval_env=eval_env,
        agent=agent,
        save_path=save_path,
        name_prefix="test_run",
        render_freq=1,
        n_eval_episodes=1
    )
    
    # Trigger recording
    callback.record_video(suffix="_fixed")
    
    # Check if file exists
    expected_file = os.path.join(save_path, "test_run_fixed.gif")
    if os.path.exists(expected_file):
        print(f"Success: {expected_file} generated.")
        # Optional: verify it's a valid GIF and has frames
        # os.remove(expected_file)
        # os.rmdir(save_path)
    else:
        print(f"Failure: {expected_file} not found.")
        sys.exit(1)

if __name__ == "__main__":
    test_video_callback()
