"""
Evalúa un modelo PPO v4 (SB3) en defend_the_center.

Uso:
    PYTHONPATH=src python -m doom_agent.algorithms.ppo_v4.eval_defend_the_center --model checkpoints/defend_the_center/ppo_v4_defend_final.zip
"""
import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from doom_agent.algorithms.ppo_v4.envs import DoomCorridorEnv, defend_actions


def build_env(window_visible: bool, frame_skip: int):
    return VecTransposeImage(DummyVecEnv([lambda: DoomCorridorEnv(
        scenario="defend_the_center.cfg",
        frame_skip=frame_skip,
        frame_size=(160, 120),
        window_visible=window_visible,
        actions=defend_actions(),
    )]))


def run_eval(model_path: Path, episodes: int, render_window: bool, frame_skip: int, sleep_time: float):
    env = build_env(window_visible=render_window, frame_skip=frame_skip)
    model = PPO.load(model_path, env=env, device="cuda" if render_window else "auto")

    rewards = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            if render_window and sleep_time > 0:
                time.sleep(sleep_time)
        rewards.append(ep_reward)
        print(f"[Eval] Episodio {ep+1}/{episodes} reward={ep_reward:.2f}")

    env.close()
    print(f"[Eval] Reward medio: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Ruta al .zip de SB3")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render-window", action="store_true", help="Mostrar ventana de VizDoom (requiere X/GUI)")
    parser.add_argument("--frame-skip", type=int, default=3, help="Frame skip para la evaluación")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay en segundos entre acciones cuando se renderiza")
    args = parser.parse_args()

    run_eval(Path(args.model), args.episodes, args.render_window, args.frame_skip, args.sleep)


if __name__ == "__main__":
    main()
