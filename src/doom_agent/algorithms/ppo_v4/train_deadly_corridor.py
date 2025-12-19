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
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv

from doom_agent.algorithms.ppo_v4.envs import DoomCorridorEnv


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
    return lambda: DoomCorridorEnv(
        scenario="deadly_corridor.cfg",
        frame_skip=stage.frame_skip,
        frame_size=(160, 120),
        doom_skill=stage.doom_skill,
        living_reward=stage.living_reward,
        window_visible=window_visible,
    )


def build_envs(n_envs: int, stage: Stage, window_visible: bool = False):
    return VecTransposeImage(DummyVecEnv([make_env_fn(stage, window_visible) for _ in range(n_envs)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Ruta a un checkpoint .zip de SB3 para reanudar (ppo_v4_*_last.zip, etc.)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / "runs" / "deadly_corridor" / run_id
    ckpt_dir = base_dir / "checkpoints" / "deadly_corridor" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.resume, device="auto") if args.resume else None
    total_trained = 0

    for stage in CURRICULUM:
        print(
            f"[Curriculum] Etapa {stage.name} skill={stage.doom_skill} "
            f"living_reward={stage.living_reward} frame_skip={stage.frame_skip} "
            f"timesteps={stage.timesteps:,} (acum {total_trained:,})"
        )
        train_env = build_envs(n_envs=16, stage=stage)
        eval_env = build_envs(n_envs=1, stage=stage)

        if model is None:
            model = PPO(
                policy="CnnPolicy",
                env=train_env,
                n_steps=2048,
                batch_size=256,
                learning_rate=2.5e-4,
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

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / stage.name),
            log_path=str(log_dir / "eval" / stage.name),
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=str(ckpt_dir / stage.name),
            name_prefix=f"ppo_v4_{stage.name}",
        )

        model.learn(
            total_timesteps=stage.timesteps,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False,
        )
        total_trained += stage.timesteps
        model.save(ckpt_dir / f"ppo_v4_{stage.name}_last")

        train_env.close()
        eval_env.close()

    model.save(ckpt_dir / "ppo_v4_final")


if __name__ == "__main__":
    main()
