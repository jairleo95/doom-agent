"""
Train PPO on VizDoom defend_the_center using Stable-Baselines3 (minimal setup).

Usage:
    PYTHONPATH=src python -m doom_agent.algorithms.ppo_v4.train_defend_the_center
    # Reanudar desde un checkpoint SB3:
    # PYTHONPATH=src python -m doom_agent.algorithms.ppo_v4.train_defend_the_center --resume path/to/ppo_v4_defend_last.zip
"""
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from doom_agent.algorithms.ppo_v4.envs import DoomCorridorEnv, defend_actions


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
    return lambda: DoomCorridorEnv(
        scenario="defend_the_center.cfg",
        frame_skip=stage.frame_skip,
        frame_size=(160, 120),
        doom_skill=stage.doom_skill,
        living_reward=stage.living_reward,
        window_visible=window_visible,
        actions=defend_actions(),  # action set específico del escenario
    )


def build_envs(n_envs: int, stage: Stage, window_visible: bool = False):
    return VecTransposeImage(DummyVecEnv([make_env_fn(stage, window_visible) for _ in range(n_envs)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Ruta a un checkpoint .zip de SB3 para reanudar (ppo_v4_defend_*_last.zip, etc.)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / "runs" / "defend_the_center" / run_id
    ckpt_dir = base_dir / "checkpoints" / "defend_the_center" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(args.resume, device="auto") if args.resume else None

    for stage in CURRICULUM:
        print(
            f"[Curriculum] Etapa {stage.name} skill={stage.doom_skill} "
            f"living_reward={stage.living_reward} frame_skip={stage.frame_skip} "
            f"timesteps={stage.timesteps:,}"
        )
        train_env = build_envs(n_envs=8, stage=stage)
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
        model.save(ckpt_dir / f"ppo_v4_{stage.name}_last")

        train_env.close()
        eval_env.close()

    model.save(ckpt_dir / "ppo_v4_defend_final")


if __name__ == "__main__":
    main()
