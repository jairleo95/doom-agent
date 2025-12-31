"""
Train PPO on VizDoom defend_the_center using Stable-Baselines3 (minimal setup).

Usage:
    PYTHONPATH=src python -m doom_agent.algorithms.ppo.v4.train_defend_the_center
    # Reanudar desde un checkpoint SB3:
    # PYTHONPATH=src python -m doom_agent.algorithms.ppo.v4.train_defend_the_center --resume path/to/ppo_v4_defend_last.zip
"""
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, SubprocVecEnv, VecFrameStack
from typing import Callable

from doom_agent.algorithms.ppo.v4.envs import DoomCorridorEnv, defend_actions


@dataclass(frozen=True)
class Stage:
    name: str
    timesteps: int
    doom_skill: int
    living_reward: float
    frame_skip: int = 3


# Curriculum pensado para mayor reactividad y menos indecisión al atacar.
CURRICULUM = [
    Stage(name="defend_skill2", timesteps=400_000, doom_skill=2, living_reward=-0.005, frame_skip=3),
    Stage(name="defend_skill3", timesteps=800_000, doom_skill=3, living_reward=-0.002, frame_skip=2),
    Stage(name="defend_skill4", timesteps=1_000_000, doom_skill=4, living_reward=0.0, frame_skip=2),
]


def make_env_fn(stage: Stage, window_visible: bool = False):
    def _init():
        return DoomCorridorEnv(
            scenario="defend_the_center.cfg",
            frame_skip=stage.frame_skip,
            frame_size=(160, 120),
            doom_skill=stage.doom_skill,
            living_reward=stage.living_reward,
            window_visible=window_visible,
            actions=defend_actions(),  # action set específico del escenario
            health_penalty=0.1,  # Penalizar daño recibido
            ammo_penalty=0.05,   # Penalizar spam de disparos
        )
    return _init


def build_envs(n_envs: int, stage: Stage, window_visible: bool = False, is_eval: bool = False):
    if is_eval or n_envs == 1:
        env = DummyVecEnv([make_env_fn(stage, window_visible) for _ in range(n_envs)])
    else:
        env = SubprocVecEnv([make_env_fn(stage, window_visible) for _ in range(n_envs)])
    
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


from doom_agent.algorithms.ppo.v4.callbacks import VideoRecorderCallback, OnBestVideoCallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Ruta a un checkpoint .zip de SB3 para reanudar (ppo_v4_defend_*_last.zip, etc.)",
    )
    parser.add_argument("--video-freq", type=int, default=100_000, help="Frecuencia de generación de GIF (en steps)")
    parser.add_argument("--video-on-best", action="store_true", help="Generar GIF cuando haya nuevo mejor modelo")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / "runs" / "defend_the_center" / run_id
    ckpt_dir = base_dir / "checkpoints" / "defend_the_center" / run_id
    video_dir = ckpt_dir / "videos"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.resume, device="auto") if args.resume else None

    for stage in CURRICULUM:
        print(
            f"[Curriculum] Etapa {stage.name} skill={stage.doom_skill} "
            f"living_reward={stage.living_reward} frame_skip={stage.frame_skip} "
            f"timesteps={stage.timesteps:,}"
        )
        n_envs_train = 12
        train_env = build_envs(n_envs=n_envs_train, stage=stage, is_eval=False)
        eval_env = build_envs(n_envs=1, stage=stage, is_eval=True)

        if model is None:
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
            model.set_env(train_env)

        # Video Recorder Callback
        video_callback = VideoRecorderCallback(
            eval_env=eval_env,
            render_freq=args.video_freq,
            save_path=str(video_dir),
            name_prefix=f"video_{stage.name}",
        )
        
        on_best_callback = None
        if args.video_on_best:
            on_best_callback = OnBestVideoCallback(video_callback, suffix=f"_best_{stage.name}")

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / stage.name),
            log_path=str(log_dir / "eval" / stage.name),
            eval_freq=max(20_000 // n_envs_train, 1),
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
        
        callbacks = [eval_callback, checkpoint_callback]
        if args.video_freq > 0:
            callbacks.append(video_callback)

        model.learn(
            total_timesteps=stage.timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
        )
        model.save(ckpt_dir / f"ppo_v4_{stage.name}_last")

        train_env.close()
        eval_env.close()

    model.save(ckpt_dir / "ppo_v4_defend_final")


if __name__ == "__main__":
    main()
