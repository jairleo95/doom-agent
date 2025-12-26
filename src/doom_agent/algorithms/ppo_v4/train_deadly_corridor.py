"""
Train PPO on VizDoom deadly_corridor using Stable-Baselines3, without relying on existing repo wrappers.

Usage:
    PYTHONPATH=src python -m doom_agent.algorithms.ppo_v4.train_deadly_corridor
"""
from pathlib import Path
import argparse
from dataclasses import dataclass
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv, SubprocVecEnv, VecFrameStack
from typing import Callable

from doom_agent.algorithms.ppo_v4.envs import DoomCorridorEnv, deadly_corridor_actions


@dataclass(frozen=True)
class Stage:
    name: str
    timesteps: int
    doom_skill: int
    living_reward: float
    frame_skip: int = 4


CURRICULUM = [
    Stage(name="skill2_warmup", timesteps=500_000, doom_skill=2, living_reward=-0.01, frame_skip=3),
    Stage(name="skill4_mid", timesteps=1_000_000, doom_skill=4, living_reward=0.0, frame_skip=3),
    Stage(name="skill5_target", timesteps=1_500_000, doom_skill=5, living_reward=0.0, frame_skip=2),
]


def make_env_fn(stage: Stage, window_visible: bool = False):
    def _init():
        return DoomCorridorEnv(
            scenario="deadly_corridor.cfg",
            frame_skip=stage.frame_skip,
            frame_size=(160, 120),
            doom_skill=stage.doom_skill,
            living_reward=stage.living_reward,
            window_visible=window_visible,
            actions=deadly_corridor_actions(),
            health_penalty=0.1,  # Penalizar daño (crucial en corridor)
            ammo_penalty=0.01,   # Penalizar spam (leve)
        )
    return _init


def build_envs(n_envs: int, stage: Stage, window_visible: bool = False, is_eval: bool = False):
    # Usar SubprocVecEnv para entrenamiento (paralelismo real) y DummyVecEnv para eval/debug
    if is_eval or n_envs == 1:
        # Eval generalmente usa 1 env, Dummy es más seguro/fácil
        env = DummyVecEnv([make_env_fn(stage, window_visible) for _ in range(n_envs)])
    else:
        env = SubprocVecEnv([make_env_fn(stage, window_visible) for _ in range(n_envs)])
    
    # VecFrameStack requiere canales primero o último dependiendo... SB3 maneja esto.
    # Pero VizDoomEnv devuelve (H, W, 1).
    # VecTransposeImage convierte (N, H, W, C) -> (N, C, H, W) para PyTorch.
    # VecFrameStack espera (N, H, W, C) si channels_order='last' (default) o (N, C, H, W) si 'first'.
    # Lo ideal:
    # 1. Env -> (H, W, 1)
    # 2. VecEnv -> (N, H, W, 1)
    # 3. VecFrameStack -> (N, H, W, 4) (apila en el canal)
    # 4. VecTransposeImage -> (N, 4, H, W) (para CNN PyTorch)
    
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = VecTransposeImage(env)
    return env


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule function
    """
    def func(progress_remaining: float) -> float:
        """
        Progress remaining starts from 1.0 to 0.0
        """
        return progress_remaining * initial_value
    return func


from doom_agent.algorithms.ppo_v4.callbacks import VideoRecorderCallback, OnBestVideoCallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Ruta a un checkpoint .zip de SB3 para reanudar (ppo_v4_*_last.zip, etc.)",
    )
    parser.add_argument("--video-freq", type=int, default=100_000, help="Frecuencia de generación de GIF (en steps)")
    parser.add_argument("--video-on-best", action="store_true", help="Generar GIF cuando haya nuevo mejor modelo")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / "runs" / "deadly_corridor" / run_id
    ckpt_dir = base_dir / "checkpoints" / "deadly_corridor" / run_id
    video_dir = ckpt_dir / "videos"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.resume, device="auto") if args.resume else None
    total_trained = 0

    for stage in CURRICULUM:
        print(
            f"[Curriculum] Etapa {stage.name} skill={stage.doom_skill} "
            f"living_reward={stage.living_reward} frame_skip={stage.frame_skip} "
            f"timesteps={stage.timesteps:,} (acum {total_trained:,})"
        )
        # Ajustamos numero de envs para aprovechar paralelismo
        n_envs_train = 12  # Ryzen 3700X (12/16 threads)
        train_env = build_envs(n_envs=n_envs_train, stage=stage, is_eval=False)
        eval_env = build_envs(n_envs=1, stage=stage, is_eval=True)

        if model is None:
            model = PPO(
                policy="CnnPolicy",
                env=train_env,
                n_steps=1024, # 1024 * 12 = 12288 buffer size
                batch_size=512, # RTX 3060 puede manejar batches grandes
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
            model.set_env(train_env)

        # Video Recorder Callback
        video_callback = VideoRecorderCallback(
            eval_env=eval_env,
            render_freq=args.video_freq,
            save_path=str(video_dir),
            name_prefix=f"video_{stage.name}",
        )
        
        # Callback triggered on new best model
        on_best_callback = None
        if args.video_on_best:
            on_best_callback = OnBestVideoCallback(video_callback, suffix=f"_best_{stage.name}")

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / stage.name),
            log_path=str(log_dir / "eval" / stage.name),
            eval_freq=max(20_000 // n_envs_train, 1), # Eval cada ~20k steps reales (menos frecuente)
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            callback_on_new_best=on_best_callback
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(50_000 // n_envs_train, 1),
            save_path=str(ckpt_dir / stage.name),
            name_prefix=f"ppo_v4_{stage.name}",
        )
        
        # List of callbacks
        callbacks = [eval_callback, checkpoint_callback]
        if args.video_freq > 0:
            callbacks.append(video_callback)

        model.learn(
            total_timesteps=stage.timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
        )
        total_trained += stage.timesteps
        model.save(ckpt_dir / f"ppo_v4_{stage.name}_last")

        train_env.close()
        eval_env.close()

    model.save(ckpt_dir / "ppo_v4_final")


if __name__ == "__main__":
    main()
