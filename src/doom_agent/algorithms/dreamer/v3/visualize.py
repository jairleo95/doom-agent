import argparse
import time
import numpy as np
import torch
import pathlib
from pathlib import Path

from doom_agent.algorithms.dreamer.v3.agent import DreamerV3Agent
from doom_agent.algorithms.dreamer.v3.doom_envs import DoomDreamerEnv, universal_actions, deadly_corridor_actions, defend_actions
from doom_agent.algorithms.dreamer.v3.curriculum import DEATHMATCH_CURRICULUM, DEADLY_CORRIDOR_CURRICULUM, DEFEND_CENTER_CURRICULUM, GRAND_CURRICULUM

def get_action_set(scenario):
    if scenario == 'deathmatch' or scenario == 'deathmatch_curriculum':
        return universal_actions()
    elif scenario == 'deadly_corridor':
        return deadly_corridor_actions()
    elif scenario == 'defend_the_center':
        return defend_actions()
    elif scenario == 'universal':
        return universal_actions()
    else:
        return universal_actions()

def main():
    parser = argparse.ArgumentParser(description='Visualize trained DreamerV3 agent')
    parser.add_argument('--path', type=str, required=True, help='Path to checkpoint .pt file or run directory')
    parser.add_argument('--scenario', type=str, default='deathmatch_curriculum', choices=['deathmatch', 'deathmatch_curriculum', 'deadly_corridor', 'defend_the_center', 'universal'], help='Scenario to run')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--fps', type=int, default=0, help='Max FPS (0 for uncapped)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on')
    
    args = parser.parse_args()
    
    # Resolve path
    path = Path(args.path)
    if path.is_dir():
        # Try to find the latest expert checkpoint
        checkpoints = list(path.glob('**/skill5_expert.pt'))
        if not checkpoints:
            checkpoints = list(path.glob('**/*.pt'))
        
        if not checkpoints:
            print(f"Error: No .pt checkpoints found in {path}")
            return
        
        # Sort by modification time to get latest
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        checkpoint_path = checkpoints[0]
        print(f"Auto-detected latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = path

    # Load curriculum to get stage settings (expert level by default for visualization)
    if args.scenario == "deathmatch" or args.scenario == "deathmatch_curriculum":
        curriculum = DEATHMATCH_CURRICULUM
        stage = curriculum.stages[-1] # Expert
    elif args.scenario == "deadly_corridor":
        curriculum = DEADLY_CORRIDOR_CURRICULUM
        stage = curriculum.stages[-1]
    elif args.scenario == "defend_the_center":
        curriculum = DEFEND_CENTER_CURRICULUM
        stage = curriculum.stages[-1]
    elif args.scenario == "universal":
        curriculum = GRAND_CURRICULUM
        stage = curriculum.stages[-1]
    
    actions = get_action_set(args.scenario)
    
    # Load checkpoint to detect obs_shape
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('agent_state_dict', checkpoint)
    
    # Auto-detect if it's grayscale (1 channel) or color (3 channels)
    # Match the key from the error: _wm.encoder._cnn.layers.0.weight
    # We look for any key that looks like the first layer of the encoder
    detected_channels = 1
    for key in state_dict.keys():
        if 'encoder._cnn.layers.0.weight' in key:
            shape = state_dict[key].shape
            detected_channels = shape[1] # [out_channels, in_channels, k, k]
            break
            
    print(f"Detected {detected_channels} input channels in checkpoint.")
    detected_obs_shape = (64, 64, detected_channels)

    # Initialize Environment with detected shape
    env = DoomDreamerEnv(
        scenario=curriculum.scenario if not stage.scenario else stage.scenario,
        actions=actions,
        frame_skip=stage.frame_skip,
        window_visible=True,
        doom_skill=stage.doom_skill,
        obs_shape=detected_obs_shape
    )
    
    # Auto-detect architectural scaling
    detected_dyn_deter = 512
    for key in state_dict.keys():
        if 'dynamics.W' in key:
            # W shape is [1, dyn_deter]
            detected_dyn_deter = state_dict[key].shape[1]
            break
    
    print(f"Detected dyn_deter={detected_dyn_deter} from checkpoint.")

    # Initialize Agent with detected shape
    agent_config = {
        'device': args.device,
        'action_dim': len(actions),
        'obs_shape': detected_obs_shape,
        'dyn_deter': detected_dyn_deter,
        'compile': False,
    }
    
    dummy_run_dir = Path("runs/visualize_temp")
    dummy_run_dir.mkdir(parents=True, exist_ok=True)
    
    agent = DreamerV3Agent(agent_config, dummy_run_dir)
    agent.load(checkpoint_path)
    
    print(f"\n--- Starting Visualization ({args.episodes} episodes) ---")
    print("Close the VizDoom window or press Ctrl+C to stop.")
    
    try:
        for ep in range(args.episodes):
            obs = env.reset()
            agent.reset_state()
            done = False
            total_reward = 0
            steps = 0
            start_time = time.time()
            
            while not done:
                loop_start = time.time()
                
                # DreamerV3 select_action expects (N, H, W, C)
                # env.reset() returns (H, W, C)
                action_idx = agent.select_action(obs, eval_mode=True, is_first=(steps==0))
                
                obs, reward, done = env.step(action_idx)
                total_reward += reward
                steps += 1
                
                # FPS cap
                if args.fps > 0:
                    elapsed = time.time() - loop_start
                    wait = (1.0 / args.fps) - elapsed
                    if wait > 0:
                        time.sleep(wait)
            
            duration = time.time() - start_time
            print(f"Episode {ep+1}: Reward={total_reward:.2f}, Steps={steps}, Duration={duration:.1f}s, Avg FPS={steps/duration:.1f}")
            
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
