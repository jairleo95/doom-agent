
import argparse
import os
from pathlib import Path
from typing import Callable
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv, VecFrameStack

from doom_agent.algorithms.ppo_v5.envs import DoomCorridorEnv, deadly_corridor_actions, defend_actions, universal_actions
from doom_agent.algorithms.ppo_v5.curriculum import (
    Stage, 
    Curriculum, 
    DEADLY_CORRIDOR_CURRICULUM, 
    DEFEND_CENTER_CURRICULUM,
    GRAND_CURRICULUM
)
from doom_agent.algorithms.ppo_v5.callbacks import VideoRecorderCallback, OnBestVideoCallback

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env_fn(stage: Stage, scenario_cfg: str, actions, window_visible: bool = False):
    def _init():
        return DoomCorridorEnv(
            scenario=stage.scenario or scenario_cfg, # Use stage override or base
            frame_skip=stage.frame_skip,
            frame_size=(160, 120),
            doom_skill=stage.doom_skill,
            living_reward=stage.living_reward,
            window_visible=window_visible,
            actions=actions,
            health_penalty=stage.health_penalty,
            ammo_penalty=stage.ammo_penalty
        )
    return _init

def build_envs(n_envs: int, stage: Stage, scenario_cfg: str, actions, window_visible: bool = False, is_eval: bool = False):
    if is_eval or n_envs == 1:
        env = DummyVecEnv([make_env_fn(stage, scenario_cfg, actions, window_visible) for _ in range(n_envs)])
    else:
        env = SubprocVecEnv([make_env_fn(stage, scenario_cfg, actions, window_visible) for _ in range(n_envs)])
    
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = VecTransposeImage(env)
    return env

def main():
    parser = argparse.ArgumentParser(description="PPO v5 Training with Sequential Curriculum")
    parser.add_argument("--scenario", type=str, required=True, choices=["deadly_corridor", "defend_the_center", "universal"], help="Scenario to train")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume (.zip). Overrides curriculum start if starting from stage 0.")
    parser.add_argument("--start-stage", type=int, default=0, help="Stage index to start from (0-based).")
    parser.add_argument("--video-freq", type=int, default=100_000, help="Video generation frequency")
    parser.add_argument("--video-on-best", action="store_true", default=True, help="Record video on new best model") # Default true for v5
    parser.add_argument("--n-envs", type=int, default=12, help="Number of training environments")
    args = parser.parse_args()

    # Select Curriculum
    if args.scenario == "deadly_corridor":
        curriculum = DEADLY_CORRIDOR_CURRICULUM
        actions = deadly_corridor_actions()
    elif args.scenario == "defend_the_center":
        curriculum = DEFEND_CENTER_CURRICULUM
        actions = defend_actions()
    elif args.scenario == "universal":
        curriculum = GRAND_CURRICULUM
        actions = universal_actions()
    else:
        raise ValueError("Unknown scenario")

    # Setup Paths
    base_dir = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / "runs" / args.scenario / run_id
    ckpt_dir = base_dir / "checkpoints" / args.scenario / run_id
    video_dir = ckpt_dir / "videos"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting PPO v5 Training: {curriculum.name}")
    print(f"Total Stages: {len(curriculum.stages)}")

    last_model_path = args.resume

    for idx, stage in enumerate(curriculum.stages):
        if idx < args.start_stage:
            print(f"Skipping Stage {idx}: {stage.name}")
            continue

        print(f"\n=== Running Stage {idx}: {stage.name} ===")
        print(f"Config: Skill={stage.doom_skill}, Reward={stage.living_reward}, HP_Pen={stage.health_penalty}, Ammo_Pen={stage.ammo_penalty}")
        
        # Build Envs
        train_env = build_envs(args.n_envs, stage, curriculum.scenario, actions, is_eval=False)
        eval_env = build_envs(1, stage, curriculum.scenario, actions, is_eval=True)

        # Initialize or Load Model
        if idx == 0 and last_model_path is None:
            print("Initializing NEW model...")
            model = PPO(
                policy="CnnPolicy",
                env=train_env,
                n_steps=1024,
                batch_size=512,
                learning_rate=linear_schedule(2.5e-4),
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.1,
                ent_coef=0.01,
                vf_coef=0.5,
                n_epochs=4,
                tensorboard_log=str(log_dir),
                verbose=1,
            )
        else:
            # WARM START
            load_path = last_model_path
            # If we are starting from stage > 0 but have no resume path, implies we should have kept previous stage model in memory?
            # Or we expect to load from file. 
            # In this loop, 'model' variable holds the object. 
            # BUT we need to set the NEW env.
            
            if 'model' in locals() and model is not None:
                print("Continuing with current model compliant with new stage...")
                model.set_env(train_env)
                # Ensure learning rate schedule is reset? 
                # Ideally, we want schedule to restart or continue? 
                # Usually schedule depends on total_timesteps of learn() call.
                # So calling learn() again resets progress_remaining for THAT call. 
                # Perfect.
            elif load_path:
                print(f"Loading Warm-Start model from: {load_path}")
                model = PPO.load(load_path, env=train_env, device="auto", tensorboard_log=str(log_dir))
            else:
                # Should not happen if logic is correct
                print("Error: Starting mid-curriculum without a model!")
                return
        
        # Callbacks
        callbacks = []
        
        # Video
        video_rec = VideoRecorderCallback(
            eval_env=eval_env,
            render_freq=args.video_freq,
            save_path=str(video_dir),
            name_prefix=f"v5_{stage.name}",
        )
        if args.video_freq > 0:
            callbacks.append(video_rec)
            
        on_best = None
        if args.video_on_best:
            on_best = OnBestVideoCallback(video_rec, suffix=f"_best_{stage.name}")
            
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / stage.name),
            log_path=str(log_dir / "eval" / stage.name),
            eval_freq=max(20_000 // args.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            callback_on_new_best=on_best
        )
        callbacks.append(eval_callback)
        
        ckpt_callback = CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=str(ckpt_dir / stage.name),
            name_prefix=f"v5_{stage.name}"
        )
        callbacks.append(ckpt_callback)

        # Train
        model.learn(
            total_timesteps=stage.timesteps,
            callback=callbacks,
            reset_num_timesteps=False 
        )
        
        # Save Last Model for Next Stage
        last_model_path = str(ckpt_dir / f"v5_{stage.name}_final.zip")
        model.save(last_model_path)
        print(f"Stage {stage.name} Complete. Saved to {last_model_path}")
        
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    main()
