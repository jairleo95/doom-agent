import torch
import time

from ppo_vizdoom_deadly_corridor import (
    ActorCritic,
    create_doom_game,
    get_deadly_corridor_actions,
    init_frame_stack,
    update_frame_stack,
    FRAME_STACK,
    DEVICE,
    BEST_MODEL_NAME,
)


def run_eval(
    num_episodes=100,
    render=True,
    model_path=f"checkpoints/deadly_corridor/{BEST_MODEL_NAME}",
):
    """
    Carga el modelo entrenado y juega algunos episodios para evaluar la política.
    """
    actions = get_deadly_corridor_actions()
    n_actions = len(actions)

    net = ActorCritic(FRAME_STACK, n_actions).to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.eval()

    game = create_doom_game("deadly_corridor.cfg", visible=render)

    episode_rewards = []
    for ep in range(1, num_episodes + 1):
        game.new_episode()
        state, frame_stack = init_frame_stack(game)

        done = False
        ep_reward = 0.0
        while not done:
            ##add time delay for better visualization
            time.sleep(0.5)

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

        episode_rewards.append(ep_reward)
        print(f"[Eval] Episodio {ep}/{num_episodes} | Reward: {ep_reward:.2f}")

    game.close()

    if episode_rewards:
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"[Eval] Reward medio sobre {num_episodes} episodios: {mean_reward:.2f}")
    else:
        print("[Eval] No se completaron episodios.")


if __name__ == "__main__":
    run_eval(num_episodes=1, render=True)
