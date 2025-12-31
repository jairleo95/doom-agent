"""
Evalúa un modelo PPO v3 (impl propia) en deadly_corridor.

Uso:
    PYTHONPATH=src python -m doom_agent.algorithms.ppo.v3.eval_deadly_corridor --model checkpoints/deadly_corridor/best.pth
"""
import argparse
import time
from pathlib import Path

import torch

from doom_agent.algorithms.ppo.v3.deadly_corridor import (
    ActorCritic,
    CKPT_CFG,
    FRAME_STACK,
    DEVICE,
    create_doom_game,
    get_deadly_corridor_actions,
    init_frame_stack,
    update_frame_stack,
)


def run_eval(model_path: Path, episodes: int, render: bool, sleep_time: float):
    actions = get_deadly_corridor_actions()
    n_actions = len(actions)

    net = ActorCritic(FRAME_STACK, n_actions).to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()

    game = create_doom_game("deadly_corridor.cfg", visible=render)

    rewards = []
    for ep in range(episodes):
        game.new_episode()
        state, frame_stack = init_frame_stack(game)

        done = False
        ep_reward = 0.0
        while not done:
            if render and sleep_time > 0:
                time.sleep(sleep_time)

            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits, _ = net(state_t)
                probs = torch.softmax(logits, dim=-1)
                action_idx = torch.argmax(probs, dim=-1).item()

            action = actions[action_idx]
            reward_env = game.make_action(action)
            done = game.is_episode_finished()
            ep_reward += reward_env

            if not done:
                state, frame_stack = update_frame_stack(frame_stack, game)

        rewards.append(ep_reward)
        print(f"[Eval] Episodio {ep+1}/{episodes} | Reward: {ep_reward:.2f}")

    game.close()
    if rewards:
        mean_r = sum(rewards) / len(rewards)
        print(f"[Eval] Reward medio: {mean_r:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Ruta al .pth. Por defecto usa CKPT_CFG.best_name en CKPT_CFG.directory")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Mostrar ventana de VizDoom (requiere X/GUI)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay entre acciones si se renderiza")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else Path(CKPT_CFG.directory) / CKPT_CFG.best_name
    run_eval(model_path, args.episodes, args.render, args.sleep)


if __name__ == "__main__":
    main()
