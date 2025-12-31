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
import hydra
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
except ImportError:
    wandb = None

# Disable audio at the OS level for headless environments
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['ALSOFT_DRIVERS'] = 'null'

from doom_agent.algorithms.dreamer.v3.agent import DreamerV3Agent
from doom_agent.algorithms.dreamer.v3.doom_envs import (
    DoomDreamerEnv, deathmatch_actions, deadly_corridor_actions, defend_actions, universal_actions
)
from doom_agent.algorithms.dreamer.v3.curriculum import (
    Curriculum, Stage
)
from doom_agent.algorithms.dreamer.v3.replay_buffer import ReplayBuffer
from doom_agent.algorithms.dreamer.v3.callbacks import (
    VideoRecorderCallback,
    CheckpointCallback,
    EvalCallback,
    MetricsCallback,
    ImaginationVideoCallback
)

def get_action_set(scenario):
    """Get action set for scenario."""
    if 'deathmatch' in scenario:
        return universal_actions() # Use universal for combat
    elif scenario == 'deadly_corridor':
        return deadly_corridor_actions()
    elif scenario == 'defend_the_center':
        return defend_actions()
    elif scenario == 'universal':
        return universal_actions()  # Universal set
    else:
        return universal_actions()

def flip_actions(actions_tensor):
    """Flip actions for horizontal symmetry (universal_actions set)."""
    # Mapping for universal_actions (indices: TL=4, TR=5, SL=2, SR=3 in universal but we mapped them differently in get_action_set)
    # Universal set indices: 3:TL, 4:TR, 5:TL+ATK, 6:TR+ATK, 7:SL, 8:SR, 10:FWD+TL, 11:FWD+TR
    flip_map = {3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7, 10: 11, 11: 10}
    
    # Create a lookup tensor for efficient mapping
    max_act = int(actions_tensor.max().item()) if actions_tensor.numel() > 0 else 12
    lookup = torch.arange(max(max_act + 1, 12), device=actions_tensor.device)
    for k, v in flip_map.items():
        if k < len(lookup):
            lookup[k] = v
            
    return lookup[actions_tensor]

def format_time(seconds):
    """Format seconds to HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def save_config(cfg, curriculum: Curriculum, log_dir: Path):
    """Save run configuration to JSON."""
    config = {
        "cfg": OmegaConf.to_container(cfg, resolve=True),
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
        obs_shape=(64, 64, 3)
    )

def update_manifest(run_id, cfg, curriculum_name, log_dir):
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
            cfg.scenario.name, 
            curriculum_name, 
            "DreamerV3",
            "RSSM",
            str(log_dir)
        ])

def save_wandb_artifact(file_path, artifact_name, artifact_type, description=None, metadata=None):
    """Upload a file to W&B Artifacts."""
    if wandb and wandb.run:
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description,
            metadata=metadata
        )
        artifact.add_file(str(file_path))
        wandb.log_artifact(artifact)
        print(f"Uploaded W&B Artifact: {artifact_name}")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train_hydra(cfg: DictConfig):
    # Select Curriculum from Hydra config
    stages = []
    for s_cfg in cfg.scenario.curriculum.stages:
        stages.append(Stage(**s_cfg))
    
    curriculum = Curriculum(
        name=cfg.scenario.name,
        scenario=cfg.scenario.scenario_name + ".cfg",
        stages=stages
    )
    
    actions = get_action_set(cfg.scenario.name)
    
    # Setup Paths
    base_dir = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_dreamer"
    log_dir = base_dir / "runs" / cfg.scenario.name / run_id
    ckpt_dir = base_dir / "checkpoints" / cfg.scenario.name / run_id
    video_dir = ckpt_dir / "videos"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment Tracking
    save_config(cfg, curriculum, log_dir)
    update_manifest(run_id, cfg, curriculum.name, log_dir)
    
    # Weights & Biases Initialization
    if wandb and cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name or run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode
        )
    
    print(f"Starting Dreamer V3 Training: {curriculum.name}")
    print(f"Total Stages: {len(curriculum.stages)}")
    print(f"Device: {cfg.device}")
    
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        print("TensorCore optimization 'high' enabled.")
    
    # Agent Config (Derived from Hydra config)
    # Convert DictConfig to plain dict for adapter
    agent_config = OmegaConf.to_container(cfg.agent, resolve=True)
    agent_config['num_actions'] = len(actions)
    agent_config['action_dim'] = len(actions)
    
    # Summary of Active Configuration
    print("\n--- Active Configuration ---")
    print(f"  Parallel Envs (n_envs): {cfg.agent.n_envs}")
    print(f"  Batch Size: {cfg.agent.batch_size}")
    print(f"  Batch Length: {cfg.agent.batch_length}")
    print(f"  Train Every: {cfg.agent.train_every}")
    print(f"  Torch Compile: {cfg.agent.get('compile', False)}")
    print(f"  Precision: {cfg.agent.get('precision', 32)}")
    print("----------------------------\n")
    
    # Initialize Agent
    agent = DreamerV3Agent(agent_config, run_dir=log_dir)
    
    # Load Resume
    if cfg.resume:
        print(f"Loading checkpoint from: {cfg.resume}")
        agent.load(cfg.resume)
        
    # Replay Buffer
    replay_buffer = ReplayBuffer(
        capacity=cfg.agent.get('buffer_capacity', 1_000_000),
        sequence_length=agent_config['batch_length'],
        obs_shape=tuple(agent_config['obs_shape'])
    )
    
    # Calculate total curriculum steps for ETA
    total_curriculum_steps = sum(s.timesteps for s in curriculum.stages)
    global_step = 0
    best_eval_reward = -float('inf')
    
    for idx, stage in enumerate(curriculum.stages):
        if idx < cfg.start_stage:
            print(f"Skipping Stage {idx}: {stage.name}")
            continue
            
        print(f"\n=== Running Stage {idx}: {stage.name} ===")
        print(f"Config: Skill={stage.doom_skill}, Reward={stage.living_reward}, Timesteps={stage.timesteps}")
        
        stage_start_time = time.time()
        
        # Envs for Stage
        scenario_cfg = stage.scenario or curriculum.scenario
        
        if cfg.agent.n_envs > 1:
            print(f"Initializing {cfg.agent.n_envs} parallel environments...")
            from doom_agent.algorithms.dreamer.v3.parallel_fix import Parallel
            from functools import partial
            train_envs = [Parallel(partial(make_env, i, scenario_cfg, actions, stage, cfg.visualize), "process") for i in range(cfg.agent.n_envs)]
        else:
            print("Initializing single environment (Check hardware config if this is unexpected!)")
            from doom_agent.algorithms.dreamer.v3.parallel_fix import Damy
            train_envs = [Damy(make_env(0, scenario_cfg, actions, stage, cfg.visualize))]
        
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
            obs_shape=(64, 64, 3)
        )
        
        # Callbacks
        stage_ckpt_dir = ckpt_dir / stage.name
        stage_ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_callback = MetricsCallback(log_path=str(log_dir / stage.name), name='metrics')
        
        video_rec = VideoRecorderCallback(
            eval_env=eval_env,
            agent=agent,
            save_path=str(video_dir / stage.name),
            name_prefix=f"dreamer_{stage.name}",
            render_freq=cfg.video_freq,
            n_eval_episodes=1,
            deterministic=True,
        )
        
        eval_callback = EvalCallback(
            eval_env=eval_env,
            agent=agent,
            eval_freq=10, # Evaluate every 10 episodes
            n_eval_episodes=3,
            callback_on_new_best=video_rec.record_video
        )
        
        # Imagination Video Logging Setup
        imag_video_rec = ImaginationVideoCallback(
            agent=agent,
            log_dir=log_dir / stage.name, # Use per-stage log dir for TB
            render_freq=cfg.video_freq or 1000
        )
        last_imag_log_step = global_step

        
        # Prefill if needed (Stage 0 only)
        if idx == 0 and not cfg.resume and global_step == 0:
            print(f"Prefilling buffer with {cfg.agent.prefill_steps} steps...")
            obs_list = [e.reset()() for e in train_envs]
            agent.reset_state()
            is_first_list = [True] * cfg.agent.n_envs
            
            steps_done = 0
            while steps_done < cfg.agent.prefill_steps:
                actions_vec = [np.random.randint(0, len(actions)) for _ in range(cfg.agent.n_envs)]
                
                step_results = [e.step(a)() for e, a in zip(train_envs, actions_vec)]
                
                for i, (next_obs, reward, done) in enumerate(step_results):
                    replay_buffer.add(obs_list[i], actions_vec[i], reward, float(done), is_first_list[i])
                    obs_list[i] = next_obs
                    is_first_list[i] = done
                    if done:
                        obs_list[i] = train_envs[i].reset()()
                        is_first_list[i] = True
                
                steps_done += cfg.agent.n_envs
                if steps_done % 1000 == 0:
                    print(f"Prefilled {steps_done}/{cfg.agent.prefill_steps} steps...")
            
            agent.reset_state()

        # Training Loop - PIPELINED to overlap CPU (Env) and GPU (Train)
        stage_step = 0
        obs_list = [e.reset()() for e in train_envs]
        agent.reset_state()
        is_first_list = [True] * cfg.agent.n_envs
        episode_count = 0
        
        env_episode_rewards = [0.0] * cfg.agent.n_envs
        env_episode_lengths = [0] * cfg.agent.n_envs
        env_episode_start_times = [time.time()] * cfg.agent.n_envs
        
        last_log_time = time.time()
        last_log_step = global_step
        last_save_step = global_step
        
        train_counter = 0
        first_train = True
        
        last_eval_time = time.time()
        last_eval_step = global_step
        stable_eta_str = "N/A"
        ema_fps = None
        
        print(f"Main pipelined training loop started. Logging every 100 steps.")
        
        # WARMUP: Start the first collection task immediately
        obs_batch = np.stack(obs_list)
        actions_vec = agent.select_action(obs_batch, is_first=is_first_list)
        if cfg.agent.n_envs == 1: actions_vec = [actions_vec]
        step_futures = [e.step(a) for e, a in zip(train_envs, actions_vec)]
        
        while stage_step < stage.timesteps:
            # ---------------------------------------------------------
            # 1. OPTIONAL: Start TRAINING Step (GPU)
            # This happens in parallel with the env.step processing in step_futures
            # ---------------------------------------------------------
            train_metrics = None
            if train_counter >= cfg.agent.train_every and len(replay_buffer) > cfg.agent.batch_size * cfg.agent.batch_length:
                if first_train:
                    print("First training step started (Benchmarking)...", flush=True)
                
                num_batches = (train_counter // cfg.agent.train_every) * cfg.agent.train_steps
                train_counter = train_counter % cfg.agent.train_every
                
                # We do the actual training. PyTorch handles GPU asynchrony.
                for _ in range(num_batches):
                    do_flip = np.random.random() < 0.5
                    batch = replay_buffer.sample(cfg.agent.batch_size, horizontal_flip=do_flip)
                    if batch:
                        if do_flip: batch['action'] = flip_actions(batch['action'])
                        train_metrics = agent.train_step(batch) # GPU Bound
                        
                        if imag_video_rec and global_step >= last_imag_log_step + imag_video_rec.render_freq:
                            imag_video_rec.record_imagination(global_step, batch)
                            last_imag_log_step = global_step
                        
                        if first_train:
                            print("First training completed!", flush=True)
                            first_train = False
                        
                        if train_metrics:
                            metrics_callback.log_training(global_step, **train_metrics)

            # ---------------------------------------------------------
            # 2. WAIT for Environment Collection (CPU/IPC)
            # ---------------------------------------------------------
            step_results = [f() for f in step_futures]
            
            # Process results from the steps just finished
            for i, (next_obs, reward, done) in enumerate(step_results):
                replay_buffer.add(obs_list[i], actions_vec[i], reward, float(done), is_first_list[i])
                
                env_episode_rewards[i] += reward
                env_episode_lengths[i] += 1
                
                obs_list[i] = next_obs
                is_first_list[i] = done
                
                if done:
                    episode_count += 1
                    ep_duration = time.time() - env_episode_start_times[i]
                    
                    info = {
                        'frags': getattr(train_envs[i], 'last_frag_count', 0),
                        'health': getattr(train_envs[i], 'last_health', 0),
                        'ammo': getattr(train_envs[i], 'last_ammo', 0)
                    }
                    if hasattr(info['frags'], '__call__'): info['frags'] = info['frags']()
                    if hasattr(info['health'], '__call__'): info['health'] = info['health']()
                    if hasattr(info['ammo'], '__call__'): info['ammo'] = info['ammo']()
                    
                    metrics_callback.log_episode(episode_count, env_episode_rewards[i], env_episode_lengths[i], ep_duration, step=global_step, info=info)
                    
                    if eval_callback.should_evaluate(episode_count):
                        eval_results = eval_callback.evaluate(global_step)
                        curr_time = time.time()
                        eval_lap_time = curr_time - last_eval_time
                        eval_lap_steps = global_step - last_eval_step
                        
                        if eval_lap_time > 0:
                            stable_fps = eval_lap_steps / eval_lap_time
                            ema_fps = stable_fps if ema_fps is None else 0.3 * stable_fps + 0.7 * ema_fps
                            stable_eta_str = format_time((total_curriculum_steps - global_step) / ema_fps)
                        
                        last_eval_time, last_eval_step = curr_time, global_step
                        metrics_callback.log_training(global_step, eval_mean_reward=eval_results['mean_reward'], eval_mean_length=eval_results['mean_length'])
                        
                        if cfg.wandb.enabled and cfg.wandb.save_artifacts and eval_results['mean_reward'] > best_eval_reward:
                            best_eval_reward = eval_results['mean_reward']
                            best_path = ckpt_dir / "best_model.pt"
                            agent.save(str(best_path))
                            save_wandb_artifact(best_path, f"{run_id}_best_model", "model")
                        
                        if eval_lap_time > 0:
                            print(f"  Lap Stats: FPS={stable_fps:.2f}, EMA FPS={ema_fps:.2f}, ETA={stable_eta_str}")

                    obs_list[i] = train_envs[i].reset()()
                    is_first_list[i] = True
                    env_episode_rewards[i], env_episode_lengths[i] = 0.0, 0
                    env_episode_start_times[i] = time.time()
            
            # ---------------------------------------------------------
            # 3. Start NEXT Collection Cycle (non-blocking start)
            # ---------------------------------------------------------
            stage_step += cfg.agent.n_envs
            global_step += cfg.agent.n_envs
            train_counter += cfg.agent.n_envs
            
            # Select next actions
            obs_batch = np.stack(obs_list)
            actions_vec = agent.select_action(obs_batch, is_first=is_first_list)
            if cfg.agent.n_envs == 1: actions_vec = [actions_vec]
            
            # Launch future step
            step_futures = [e.step(a) for e, a in zip(train_envs, actions_vec)]

            # ---------------------------------------------------------
            # 4. Progress Logging & Maintenance
            # ---------------------------------------------------------
            if global_step >= last_log_step + 100:
                current_time = time.time()
                time_diff = current_time - last_log_time
                fps = (global_step - last_log_step) / time_diff if time_diff > 0 else 0
                print(f"[{stage.name}] Step {stage_step}/{stage.timesteps} ({ (stage_step/stage.timesteps)*100:.1f}%) - FPS: {fps:.2f}")
                metrics_callback.log_training(global_step, fps=fps)
                last_log_time, last_log_step = current_time, global_step
            
            if video_rec.should_record(global_step):
                video_rec.record_video(suffix=f"_step_{global_step}")
                
            if global_step % 50_000 == 0:
                agent.save(str(stage_ckpt_dir / f"dreamer_{stage.name}_{global_step}.pt"))
        
        # Stage Complete
        agent.save(str(stage_ckpt_dir / f"dreamer_{stage.name}_final.pt"))
        print(f"Stage {stage.name} Complete.")
        
        for e in train_envs: e.close()
        eval_env.close()
        metrics_callback.save()

    if wandb and cfg.wandb.enabled:
        wandb.finish()
    print("Training Complete.")

if __name__ == "__main__":
    train_hydra()
