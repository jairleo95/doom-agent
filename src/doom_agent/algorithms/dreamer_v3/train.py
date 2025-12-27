"""
Dreamer V3 Training Script (Refactored to match PPO v5)
"""

import argparse
import os
import json
import csv
import time
import sys
from pathlib import Path
# Allow importing nm512_dreamer from local dir
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from datetime import datetime
from dataclasses import asdict

import torch

# Disable audio at the OS level for headless environments
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['ALSOFT_DRIVERS'] = 'null'

from doom_agent.algorithms.dreamer_v3.agent import DreamerV3Agent
from doom_agent.algorithms.dreamer_v3.doom_envs import (
    DoomDreamerEnv, deathmatch_actions, deadly_corridor_actions, defend_actions, universal_actions
)
from doom_agent.algorithms.dreamer_v3.curriculum import (
    DEATHMATCH_CURRICULUM,
    DEADLY_CORRIDOR_CURRICULUM,
    DEFEND_CENTER_CURRICULUM,
    GRAND_CURRICULUM,
    Curriculum
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
        return universal_actions() # Use universal for combat
    elif scenario == 'deadly_corridor':
        return deadly_corridor_actions()
    elif scenario == 'defend_the_center':
        return defend_actions()
    elif scenario == 'universal':
        return universal_actions()  # Universal set
    else:
        return universal_actions()

def save_config(args, curriculum: Curriculum, log_dir: Path):
    """Save run configuration to JSON."""
    config = {
        "args": vars(args),
        "curriculum": {
            "name": curriculum.name,
            "scenario": curriculum.scenario,
            "stages": [asdict(s) for s in curriculum.stages]
        },
        "timestamp": datetime.now().isoformat()
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

def make_env(idx, scenario_cfg, actions, stage_config, visualize=False):
    """Top-level factory for pickling reliability."""
    return DoomDreamerEnv(
        scenario=scenario_cfg,
        actions=actions,
        frame_skip=stage_config.frame_skip,
        window_visible=visualize if (visualize and idx == 0) else False,
        doom_skill=stage_config.doom_skill,
        living_reward=stage_config.living_reward,
        health_penalty=stage_config.health_penalty,
        ammo_penalty=stage_config.ammo_penalty,
        frag_bonus=stage_config.frag_bonus,
        obs_shape=(64, 64, 1)
    )

def update_manifest(run_id, args, curriculum_name, log_dir):
    """Append run to a master CSV manifest."""
    manifest_path = log_dir.parent.parent / "experiments_manifest.csv"
    file_exists = manifest_path.exists()
    
    with open(manifest_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["RunID", "Date", "Scenario", "Curriculum", "Policy", "Model", "LogDir"])
        
        writer.writerow([
            run_id, 
            datetime.now().isoformat(), 
            args.scenario, 
            curriculum_name, 
            "DreamerV3",
            "RSSM",
            str(log_dir)
        ])

def main():
    parser = argparse.ArgumentParser(description="Dreamer V3 Training (PPO v5 Style)")
    parser.add_argument("--scenario", type=str, required=True, 
                       choices=["deathmatch", "deadly_corridor", "defend_the_center", "universal"],
                       help="Scenario to train")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g. .pt file)")
    parser.add_argument("--start-stage", type=int, default=0, help="Stage index to start from (0-based)")
    
    # Common PPO v5 args alignment
    parser.add_argument("--video-freq", type=int, default=50_000, help="Video recording frequency (steps)")
    parser.add_argument("--video-on-best", action="store_true", default=True, help="Record video on best model (handled by Eval callback)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs (Dreamer usually uses 1 but can process batched)")
    
    # Dreamer specific
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true", help="Show VizDoom game window during training")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--batch-length", type=int, default=50, help="Sequence length (Batch Length)")
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000, help="Replay buffer capacity")
    parser.add_argument("--train-every", type=int, default=5, help="Train every N steps")
    parser.add_argument("--train-steps", type=int, default=1, help="Gradient steps per train_every")
    parser.add_argument("--prefill-steps", type=int, default=5000, help="Random steps to prefill buffer")
    
    
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
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_dreamer"
    log_dir = base_dir / "runs" / args.scenario / run_id
    ckpt_dir = base_dir / "checkpoints" / args.scenario / run_id
    video_dir = ckpt_dir / "videos"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment Tracking
    save_config(args, curriculum, log_dir)
    update_manifest(run_id, args, curriculum.name, log_dir)
    
    print(f"Starting Dreamer V3 Training: {curriculum.name}")
    print(f"Total Stages: {len(curriculum.stages)}")
    print(f"Device: {args.device}")
    
    # Agent Config
    config = {
        'batch_size': args.batch_size,
        'batch_length': args.batch_length,
        'device': args.device,
        'obs_shape': (64, 64, 1), 
        'action_dim': len(actions),
        'num_actions': len(actions),
        'compile': False, # Disable torch.compile to avoid nvcc PermissionError
        'precision': 32,
    }
    
    # Initialize Agent
    agent = DreamerV3Agent(config, run_dir=log_dir)
    
    # Load Resume
    if args.resume:
        print(f"Loading checkpoint from: {args.resume}")
        agent.load(args.resume)
        
    # Replay Buffer
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        sequence_length=config['batch_length']
    )
    
    
    global_step = 0
    
    for idx, stage in enumerate(curriculum.stages):
        if idx < args.start_stage:
            print(f"Skipping Stage {idx}: {stage.name}")
            continue
            
        print(f"\n=== Running Stage {idx}: {stage.name} ===")
        print(f"Config: Skill={stage.doom_skill}, Reward={stage.living_reward}, Timesteps={stage.timesteps}")
        
        stage_start_time = time.time()
        
        # Envs for Stage
        scenario_cfg = stage.scenario or curriculum.scenario
        
        if args.n_envs > 1:
            print(f"Initializing {args.n_envs} parallel environments...")
            from doom_agent.algorithms.dreamer_v3.parallel_fix import Parallel
            from functools import partial
            train_envs = [Parallel(partial(make_env, i, scenario_cfg, actions, stage, args.visualize), "process") for i in range(args.n_envs)]
        else:
            print("Initializing single environment...")
            from doom_agent.algorithms.dreamer_v3.parallel_fix import Damy
            train_envs = [Damy(make_env(0, scenario_cfg, actions, stage, args.visualize))]
        
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
            obs_shape=(64, 64, 1)
        )
        
        # Callbacks
        stage_ckpt_dir = ckpt_dir / stage.name
        stage_ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # We simulate PPO callbacks logic manually mostly
        metrics_callback = MetricsCallback(log_path=str(log_dir / stage.name), name='metrics')
        
        video_rec = VideoRecorderCallback(
            eval_env=eval_env,
            agent=agent,
            save_path=str(video_dir / stage.name),
            name_prefix=f"dreamer_{stage.name}",
            render_freq=args.video_freq,
            n_eval_episodes=1,
            deterministic=True,
        )
        
        eval_callback = EvalCallback(
            eval_env=eval_env,
            agent=agent,
            eval_freq=10, # Evaluate every 10 episodes
            n_eval_episodes=3,
            callback_on_new_best=video_rec.record_video if args.video_on_best else None
        )

        
        # Prefill if needed (Stage 0 only)
        if idx == 0 and not args.resume and global_step == 0:
            print(f"Prefilling buffer with {args.prefill_steps} steps...")
            # Parallel prefill
            obs_list = [e.reset()() for e in train_envs]
            agent.reset_state()
            is_first_list = [True] * args.n_envs
            
            steps_done = 0
            while steps_done < args.prefill_steps:
                actions_vec = [np.random.randint(0, len(actions)) for _ in range(args.n_envs)]
                
                step_results = [e.step(a)() for e, a in zip(train_envs, actions_vec)]
                
                for i, (next_obs, reward, done) in enumerate(step_results):
                    replay_buffer.add(obs_list[i], actions_vec[i], reward, float(done), is_first_list[i])
                    obs_list[i] = next_obs
                    is_first_list[i] = done
                    if done:
                        obs_list[i] = train_envs[i].reset()()
                        is_first_list[i] = True
                
                steps_done += args.n_envs
                if steps_done % 1000 == 0:
                    print(f"Prefilled {steps_done}/{args.prefill_steps} steps...")
            
            agent.reset_state() # Reset again for main loop

        # Training Loop
        stage_step = 0
        obs_list = [e.reset()() for e in train_envs]
        agent.reset_state()
        is_first_list = [True] * args.n_envs
        episode_count = 0
        
        # Performance tracking (vectorized)
        env_episode_rewards = [0.0] * args.n_envs
        env_episode_lengths = [0] * args.n_envs
        env_episode_start_times = [time.time()] * args.n_envs
        
        last_log_time = time.time()
        last_log_step = global_step
        
        while stage_step < stage.timesteps:
            # Action (batched)
            obs_batch = np.stack(obs_list)
            actions_vec = agent.select_action(obs_batch, is_first=is_first_list)
            
            # If single env, select_action returns single int, convert to list for consistency
            if args.n_envs == 1:
                actions_vec = [actions_vec]
            
            # Step all envs
            step_futures = [e.step(a) for e, a in zip(train_envs, actions_vec)]
            step_results = [f() for f in step_futures]
            
            for i, (next_obs, reward, done) in enumerate(step_results):
                replay_buffer.add(obs_list[i], actions_vec[i], reward, float(done), is_first_list[i])
                
                env_episode_rewards[i] += reward
                env_episode_lengths[i] += 1
                
                obs_list[i] = next_obs
                is_first_list[i] = done
                
                if done:
                    episode_count += 1
                    ep_duration = time.time() - env_episode_start_times[i]
                    # Log training episode from this env
                    metrics_callback.log_episode(episode_count, env_episode_rewards[i], env_episode_lengths[i], ep_duration, step=global_step)
                    
                    # Evaluation (on main env / eval env)
                    if eval_callback.should_evaluate(episode_count):
                        eval_results = eval_callback.evaluate(global_step)
                        metrics_callback.log_training(global_step, 
                                                     eval_mean_reward=eval_results['mean_reward'],
                                                     eval_mean_length=eval_results['mean_length'])

                    # Reset this env
                    obs_list[i] = train_envs[i].reset()()
                    is_first_list[i] = True
                    env_episode_rewards[i] = 0.0
                    env_episode_lengths[i] = 0
                    env_episode_start_times[i] = time.time()
            
            stage_step += args.n_envs
            global_step += args.n_envs
            
            # Periodic logging (FPS and training progress)
            if global_step % 100 == 0:
                current_time = time.time()
                time_diff = current_time - last_log_time
                step_diff = global_step - last_log_step
                fps = step_diff / time_diff if time_diff > 0 else 0
                
                print(f"[{stage.name}] Step {stage_step}/{stage.timesteps} (Global {global_step}) - FPS: {fps:.2f}")
                metrics_callback.log_training(global_step, fps=fps)
                
                last_log_time = current_time
                last_log_step = global_step
            
            # Train
            should_train = (global_step % args.train_every == 0) and (len(replay_buffer) > config['batch_size'] * config['batch_length'])
            if should_train:
                for _ in range(args.train_steps):
                    batch = replay_buffer.sample(args.batch_size)
                    metrics = agent.train_step(batch)
                    metrics_callback.log_training(global_step, **metrics)
            
            # Periodic logging/video/eval
            if global_step % 1000 == 0:
                print(f"Step {stage_step}/{stage.timesteps} (Global {global_step})")
            
            if video_rec.should_record(global_step):
                video_rec.record_video(suffix=f"_step_{global_step}")
                
            # Checkpoint Every 50k steps?
            if global_step % 50_000 == 0:
                path = stage_ckpt_dir / f"dreamer_{stage.name}_{global_step}.pt"
                agent.save(str(path))
        
        # Stage Complete
        duration = time.time() - stage_start_time
        final_path = stage_ckpt_dir / f"dreamer_{stage.name}_final.pt"
        agent.save(str(final_path))
        print(f"Stage {stage.name} Complete. Saved to {final_path}")
        
        # Log Result
        stage_results = {
            "stage": stage.name,
            "duration_seconds": duration,
            "timesteps": stage_step,
            "final_model": str(final_path),
            "global_step": global_step
        }
        with open(log_dir / f"result_{stage.name}.json", "w") as f:
            json.dump(stage_results, f, indent=4)
            
        for e in train_envs:
            e.close()
        eval_env.close()
        metrics_callback.save()

    print("Training Complete.")

if __name__ == "__main__":
    main()
