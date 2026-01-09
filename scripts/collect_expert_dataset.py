
import os
import sys
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Add project root and src to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

from doom_agent.algorithms.dreamer.v3.doom_envs import DoomDreamerEnv

# Imports with fallback for adapters
try:
    from scripts.arnold_adapter import ArnoldAdapter
except (ImportError, ModuleNotFoundError):
    try:
        from arnold_adapter import ArnoldAdapter
    except:
        print("❌ Could not import ArnoldAdapter. Make sure 'external/arnold' is set up.")
        sys.exit(1)

class DatasetCollector:
    def __init__(self, agent, scenario='deathmatch.cfg', save_dir='data/expert_replays', 
                 chunk_size=10, render=False):
        self.agent = agent
        self.scenario = scenario
        self.save_dir = Path(root_dir / save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.render = render
        
        self.env = DoomDreamerEnv(
            scenario=scenario,
            obs_shape=(64, 64, 3), # DreamerV3 standard resolution
            window_visible=render,
            frame_skip=4
        )
        
        print(f"💾 Data will be saved to: {self.save_dir}")

    def collect(self, total_episodes=100):
        print(f"🎬 Starting collection of {total_episodes} episodes...")
        
        # Temporary buffers
        buffer_obs = []
        buffer_actions = []
        buffer_rewards = []
        buffer_dones = []
        
        current_chunk = 0
        episodes_in_chunk = 0
        total_steps_collected = 0
        
        for ep in tqdm(range(total_episodes), desc="Collecting Data"):
            obs = self.env.reset()
            done = False
            info = {'health': 100, 'ammo': 50, 'frags': 0}
            
            # Reset agent state if possible
            if hasattr(self.agent, "reset"):
                self.agent.reset()
                
            steps_in_ep = 0
            
            while not done:
                # Get action from expert
                action_vec = self.agent.select_action(
                    obs, 
                    health=info.get('health', 100), 
                    ammo=info.get('ammo', 50),
                    frags=info.get('frags', 0)
                )
                
                # Store transition BEFORE step (obs, action)
                buffer_obs.append(obs)
                buffer_actions.append(action_vec)
                
                # Step environment
                next_obs, reward, done, info = self.env.step_manual(action_vec)
                
                # Store result (reward, done)
                buffer_rewards.append(reward)
                buffer_dones.append(done)
                
                obs = next_obs
                steps_in_ep += 1
                total_steps_collected += 1
                
                if self.render:
                    time.sleep(0.01)
            
            episodes_in_chunk += 1
            
            # Save Chunk
            if episodes_in_chunk >= self.chunk_size:
                self._save_chunk(current_chunk, buffer_obs, buffer_actions, buffer_rewards, buffer_dones)
                
                # Clear buffers
                buffer_obs = []
                buffer_actions = []
                buffer_rewards = []
                buffer_dones = []
                
                current_chunk += 1
                episodes_in_chunk = 0
        
        # Save remaining data
        if episodes_in_chunk > 0:
            self._save_chunk(current_chunk, buffer_obs, buffer_actions, buffer_rewards, buffer_dones)

        self.env.close()
        print(f"✅ Collection Complete! Total Steps: {total_steps_collected}")

    def _save_chunk(self, chunk_id, obs, actions, rewards, dones):
        filename = self.save_dir / f"chunk_{chunk_id:04d}.npz"
        
        # Convert to numpy arrays with efficient types
        np_obs = np.array(obs, dtype=np.uint8)
        np_actions = np.array(actions, dtype=np.int8) # 7-button vector
        np_rewards = np.array(rewards, dtype=np.float32)
        np_dones = np.array(dones, dtype=bool)
        
        np.savez_compressed(
            filename,
            obs=np_obs,
            actions=np_actions,
            rewards=np_rewards,
            dones=np_dones
        )
        # print(f"Saved chunk {chunk_id} ({len(np_obs)} steps) to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect expert replay dataset")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to collect")
    parser.add_argument("--chunk_size", type=int, default=10, help="Episodes per file chunk")
    parser.add_argument("--render", action="store_true", help="Visualize collection (slower)")
    
    args = parser.parse_args()
    
    print("🤖 Initializing Arnold Agent (Expert)...")
    # Using the standard path from benchmark scripts
    arnold_path = str(root_dir / "external/arnold/pretrained/vizdoom_2017_track2.pth")
    agent = ArnoldAdapter(model_path=arnold_path)
    
    collector = DatasetCollector(
        agent=agent,
        save_dir="data/expert_replays",
        chunk_size=args.chunk_size,
        render=args.render
    )
    
    collector.collect(total_episodes=args.episodes)
