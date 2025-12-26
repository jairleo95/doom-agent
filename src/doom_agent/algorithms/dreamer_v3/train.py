"""
Dreamer V3 Training Script with Curriculum Support

Main training script similar to PPO v5's structure.
Supports multi-stage curriculum training.
"""

import argparse
import os
import json
import time
from pathlib import Path
from datetime import datetime

import torch

from doom_agent.algorithms.dreamer_v3.agent import DreamerV3Agent
from doom_agent.algorithms.dreamer_v3.doom_envs import (
    DoomDreamerEnv, deathmatch_actions, deadly_corridor_actions, defend_actions
)
from doom_agent.algorithms.dreamer_v3.curriculum import (
    DEATHMATCH_CURRICULUM,
    DEADLY_CORRIDOR_CURRICULUM,
    DEFEND_CENTER_CURRICULUM,
    GRAND_CURRICULUM
)
from doom_agent.algorithms.dreamer_v3.replay_buffer import ReplayBuffer
from doom_agent.algorithms.dreamer_v3.callbacks import (
    VideoRecorderCallback,
    CheckpointCallback,
    EvalCallback,
    MetricsCallback
)


def get_action_set(scenario):
    """Get action set for scenario."""
    if scenario == 'deathmatch':
        return deathmatch_actions()
    elif scenario == 'deadly_corridor':
        return deadly_corridor_actions()
    elif scenario == 'defend_the_center':
        return defend_actions()
    elif scenario == 'universal':
        return deathmatch_actions()  # Use deathmatch actions for universal
    else:
        return deathmatch_actions()


def save_config(args, curriculum, log_dir):
    """Save run configuration to JSON."""
    config = {
        "args": vars(args),
        "curriculum": {
            "name": curriculum.name,
            "scenario": curriculum.scenario,
            "stages": [
                {
                    "name": s.name,
                    "episodes": s.episodes,
                    "doom_skill": s.doom_skill,
                    "living_reward": s.living_reward,
                    "frame_skip": s.frame_skip,
                    "health_penalty": s.health_penalty,
                    "ammo_penalty": s.ammo_penalty,
                    "frag_bonus": s.frag_bonus,
                    "scenario": s.scenario
                }
                for s in curriculum.stages
            ]
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Dreamer V3 Training with Curriculum")
    parser.add_argument("--scenario", type=str, required=True, 
                       choices=["deathmatch", "deadly_corridor", "defend_the_center", "universal"],
                       help="Scenario to train")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start-stage", type=int, default=0, help="Stage index to start from (0-based)")
    parser.add_argument("--window-visible", action="store_true", help="Show game window")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=50, help="Sequence length")
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000, help="Replay buffer capacity")
    parser.add_argument("--prefill-steps", type=int, default=5000, help="Random exploration steps")
    parser.add_argument("--train-every", type=int, default=5, help="Train every N steps")
    parser.add_argument("--train-steps", type=int, default=1, help="Training steps per update")
    parser.add_argument("--actor-critic-every", type=int, default=5, help="Actor-critic update frequency")
    parser.add_argument("--imagination-horizon", type=int, default=15, help="Imagination horizon")
    
    # Logging and checkpointing
    parser.add_argument("--log-every", type=int, default=10, help="Log every N episodes")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N episodes")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluate every N episodes")
    parser.add_argument("--video-every", type=int, default=100, help="Record video every N episodes")
    
    args = parser.parse_args()
    
    # Select Curriculum
    if args.scenario == "deathmatch":
        curriculum = DEATHMATCH_CURRICULUM
    elif args.scenario == "deadly_corridor":
        curriculum = DEADLY_CORRIDOR_CURRICULUM
    elif args.scenario == "defend_the_center":
        curriculum = DEFEND_CENTER_CURRICULUM
    elif args.scenario == "universal":
        curriculum = GRAND_CURRICULUM
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")
    
    actions = get_action_set(args.scenario)
    
    # Setup Paths
    base_dir = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / "runs" / args.scenario / run_id
    ckpt_dir = base_dir / "checkpoints" / args.scenario / run_id
    video_dir = ckpt_dir / "videos"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Config
    save_config(args, curriculum, log_dir)
    
    print(f"Starting Dreamer V3 Training: {curriculum.name}")
    print(f"Total Stages: {len(curriculum.stages)}")
    print(f"Device: {args.device}")
    print(f"Log dir: {log_dir}")
    print(f"Checkpoint dir: {ckpt_dir}")
    
    # Agent Configuration
    config = {
        # Training
        'batch_size': args.batch_size,
        'batch_length': args.sequence_length, # NM512 uses batch_length
        'train_every': args.train_every,
        'train_steps': args.train_steps,
        'train_ratio': 512, # NM512 param: steps per batch? No, replay ratio.
        # We manually control training steps in our loop, so this might be ignorable 
        # inside the adapter unless we use agent's internal logic.
        
        'device': args.device,
        'compile': False, # Disable compilation for simplicity initially
        'precision': 32,
        
        # Environment settings required by params
        'obs_shape': (64, 64, 1), # H, W, C
        'action_dim': len(actions),
        'num_actions': len(actions),
        
        # Paths
        'logdir': str(log_dir),
    }
    
    # Initialize Agent
    # Pass run_id or log_dir to adapter
    agent = DreamerV3Agent(config, run_dir=log_dir)
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        agent.load(args.resume)
    
    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(
        capacity=config['buffer_capacity'] if 'buffer_capacity' in config else args.buffer_capacity,
        sequence_length=config['batch_length']
    )
    
    # Global counters
    global_step = 0
    global_episode = 0
    
    # Train each stage
    for stage_idx, stage in enumerate(curriculum.stages):
        if stage_idx < args.start_stage:
            print(f"\nSkipping Stage {stage_idx}: {stage.name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx}: {stage.name}")
        print(f"{'='*60}")
        print(f"Episodes: {stage.episodes}")
        # ... (print stats)
        
        stage_start_time = time.time()
        
        # Create environments for this stage
        scenario_cfg = stage.scenario or curriculum.scenario
        
        train_env = DoomDreamerEnv(
            scenario=scenario_cfg,
            actions=actions,
            frame_skip=stage.frame_skip,
            window_visible=args.window_visible,
            doom_skill=stage.doom_skill,
            living_reward=stage.living_reward,
            health_penalty=stage.health_penalty,
            ammo_penalty=stage.ammo_penalty,
            frag_bonus=stage.frag_bonus,
            obs_shape=config['obs_shape']
        )
        
        eval_env = DoomDreamerEnv(
            scenario=scenario_cfg,
            actions=actions,
            frame_skip=stage.frame_skip,
            window_visible=False,
            doom_skill=stage.doom_skill,
            living_reward=stage.living_reward,
            health_penalty=stage.health_penalty,
            ammo_penalty=stage.ammo_penalty,
            frag_bonus=stage.frag_bonus,
            obs_shape=config['obs_shape']
        )
        
        # Setup Callbacks
        stage_ckpt_dir = ckpt_dir / stage.name
        stage_log_dir = log_dir / stage.name
        stage_video_dir = video_dir / stage.name
        
        stage_ckpt_dir.mkdir(parents=True, exist_ok=True)
        stage_log_dir.mkdir(parents=True, exist_ok=True)
        stage_video_dir.mkdir(parents=True, exist_ok=True)
        
        video_callback = VideoRecorderCallback(
            eval_env=eval_env,
            agent=agent,
            save_path=str(stage_video_dir),
            name_prefix=f"dreamer_{stage.name}",
            record_freq=args.video_every
        )
        
        checkpoint_callback = CheckpointCallback(
            agent=agent,
            save_path=str(stage_ckpt_dir),
            name_prefix=f"dreamer_{stage.name}",
            save_freq=args.save_every
        )
        
        eval_callback = EvalCallback(
            eval_env=eval_env,
            agent=agent,
            eval_freq=args.eval_every,
            n_eval_episodes=5
        )
        
        metrics_callback = MetricsCallback(
            log_path=str(stage_log_dir),
            name='metrics'
        )
        
        # Prefill buffer (only on first stage)
        if stage_idx == 0 and not args.resume:
            print(f"\nPrefilling replay buffer with {args.prefill_steps} steps...")
            obs = train_env.reset()
            agent.reset_state()
            
            for step in range(args.prefill_steps):
                import numpy as np
                action = np.random.randint(0, config['action_dim'])
                next_obs, reward, done = train_env.step(action)
                
                replay_buffer.add(obs, action, reward, float(done))
                
                obs = next_obs
                if done:
                    obs = train_env.reset()
                    agent.reset_state()
                
                if (step + 1) % 1000 == 0:
                    print(f"  Prefilled {step + 1}/{args.prefill_steps} steps")
            
            print(f"Prefill complete. Buffer size: {len(replay_buffer)}")
        
        # Training loop for this stage
        print(f"\nTraining for {stage.episodes} episodes...")
        
        for episode in range(stage.episodes):
            episode_start_time = time.time()
            
            obs = train_env.reset()
            agent.reset_state()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Select action
                action = agent.select_action(obs)
                
                # Environment step
                next_obs, reward, done = train_env.step(action)
                
                # Store transition
                replay_buffer.add(obs, action, reward, float(done))
                
                episode_reward += reward
                episode_length += 1
                global_step += 1
                
                # Train agent
                if global_step % config['train_every'] == 0 and len(replay_buffer) >= config['batch_size'] * config['batch_length']:
                    for _ in range(config['train_steps']):
                        # Sample batch
                        batch = replay_buffer.sample(config['batch_size'])
                        
                        if batch is not None:
                            # Train world model & actor-critic (via adapter)
                            metrics = agent.train_step(batch)
                            
                            # Log metrics (unpack NM512 metrics)
                            # We can log all of them or filter
                            # metrics_callback.log_training(**metrics) # Need to adapt metrics names if strictly typed

                
                obs = next_obs
            
            # Episode finished
            episode_duration = time.time() - episode_start_time
            global_episode += 1
            
            # Log episode
            metrics_callback.log_episode(global_episode, episode_reward, episode_length, episode_duration)
            
            # Logging
            if (episode + 1) % args.log_every == 0:
                stats = metrics_callback.get_recent_stats(n=100)
                print(f"\nEpisode {episode + 1}/{stage.episodes} (Global: {global_episode})")
                print(f"  Steps: {global_step}")
                print(f"  Episode Reward: {episode_reward:.2f}")
                print(f"  Episode Length: {episode_length}")
                print(f"  Mean Reward (100): {stats.get('mean_reward', 0):.2f}")
                print(f"  Mean Length (100): {stats.get('mean_length', 0):.1f}")
                print(f"  Duration: {episode_duration:.2f}s")
                print(f"  Buffer Size: {len(replay_buffer)}")
            
            # Callbacks
            if checkpoint_callback.should_save(episode + 1):
                checkpoint_callback.save_checkpoint(episode + 1)
            
            if eval_callback.should_evaluate(episode + 1):
                eval_callback.evaluate(episode + 1)
            
            if video_callback.should_record(episode + 1):
                video_callback.record_video(episode + 1)
        
        # Stage complete
        stage_duration = time.time() - stage_start_time
        
        # Save final model for this stage
        final_path = stage_ckpt_dir / f"dreamer_{stage.name}_final.pt"
        agent.save(str(final_path))
        print(f"\nStage {stage.name} complete!")
        print(f"  Duration: {stage_duration / 60:.1f} minutes")
        print(f"  Saved final model: {final_path}")
        
        # Save metrics
        metrics_callback.save()
        
        # Close environments
        train_env.close()
        eval_env.close()
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total Episodes: {global_episode}")
    print(f"Total Steps: {global_step}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    main()
