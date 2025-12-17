import random
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from doom_agent.paths import scenario_path
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

# =========================================
# CONFIG
# =========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RL
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.1
LR = 2.5e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

# PPO
ROLLOUT_STEPS = 2048       # pasos por update
NUM_UPDATES = 400          # "épocas" globales de PPO
PPO_EPOCHS = 4             # epochs por batch
MINI_BATCH_SIZE = 256

# Observaciones
FRAME_STACK = 4
IMG_SIZE = (84, 84)

REWARD_SCALE = 1.0 / 25.0  # escalado de recompensa para estabilidad

# =========================================
# ENV HELPERS (ViZDoom)
# =========================================

def create_doom_game(config_path="basic.cfg"):
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(scenario_path(config_path))

    game = DoomGame()
    game.load_config(str(config_file))
    game.set_window_visible(True)  # pon True si quieres ver la ventana
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.init()
    return game


def get_basic_actions():
    # Para basic.cfg suele ser: izquierda, derecha, disparar
    return [
        [1, 0, 0],  # left
        [0, 1, 0],  # right
        [0, 0, 1],  # shoot
    ]


def preprocess_frame(frame, new_size=IMG_SIZE):
    """
    frame: (C,H,W) RGB
    return: (1,H,W) float32 [0,1]
    """
    if frame.ndim == 3 and frame.shape[0] <= 4:
        frame = np.transpose(frame, (1, 2, 0))  # (H,W,C)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[np.newaxis, :, :]


def init_frame_stack(game):
    state = game.get_state()
    frame = preprocess_frame(state.screen_buffer)   # (1,84,84)
    stack = deque([frame for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
    return stack_to_state(stack), stack


def stack_to_state(stack):
    # (FRAME_STACK,84,84)
    return np.concatenate(list(stack), axis=0)


def update_frame_stack(stack, game):
    frame = preprocess_frame(game.get_state().screen_buffer)
    stack.append(frame)
    return stack_to_state(stack), stack


# =========================================
# RED ACTOR-CRÍTICO (PPO)
# =========================================

class ActorCritic(nn.Module):
    def __init__(self, input_channels, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 84x84 -> 20x20 -> 9x9 -> 7x7
        self.fc = nn.Linear(64 * 7 * 7, 512)

        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)   # value: (B,)


def get_action_and_value(net, state):
    """
    state: np.array (C,H,W)
    """
    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits, value = net(state_t)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    action_idx = dist.sample()
    log_prob = dist.log_prob(action_idx)
    entropy = dist.entropy()

    return (
        int(action_idx.item()),
        log_prob.squeeze(0),
        entropy.squeeze(0),
        value.squeeze(0)
    )


# =========================================
# GAE + RETURNS
# =========================================

def compute_gae(rewards, dones, values, last_value, gamma, lam):
    """
    rewards: list[T]
    dones:   list[T] bool
    values:  list[T] (values en cada step)
    last_value: V(s_T) para bootstrap
    """
    T = len(rewards)
    values = np.append(values, last_value)  # len T+1
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


# =========================================
# PPO UPDATE
# =========================================

def ppo_update(net, optimizer, states, actions, old_log_probs, returns, advantages):
    """
    states:   (N,C,H,W)
    actions:  (N,)
    old_log_probs: (N,)
    returns:  (N,)
    advantages: (N,)
    """
    num_steps = states.size(0)
    indices = np.arange(num_steps)
    grad_norms = []

    for _ in range(PPO_EPOCHS):
        np.random.shuffle(indices)

        for start in range(0, num_steps, MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            mb_idx = indices[start:end]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_advantages = advantages[mb_idx]

            logits, values = net(mb_states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (mb_returns - values).pow(2).mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            grad_norms.append(grad_norm.item())
            optimizer.step()
    if grad_norms:
        return float(np.mean(grad_norms))
    return 0.0


# =========================================
# TRAIN LOOP PPO
# =========================================

def train_ppo():
    game = create_doom_game("basic.cfg")
    actions = get_basic_actions()
    n_actions = len(actions)

    net = ActorCritic(FRAME_STACK, n_actions).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    writer = SummaryWriter(log_dir="runs/vizdoom_ppo_basic")

    episode_rewards = []
    episode_raw_reward = 0.0
    global_step = 0

    # iniciar primer episodio
    game.new_episode()
    state, frame_stack = init_frame_stack(game)

    for update in range(1, NUM_UPDATES + 1):
        # buffers
        states = []
        action_idxs = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        entropies = []

        steps_collected = 0

        while steps_collected < ROLLOUT_STEPS:
            if game.is_episode_finished():
                # registrar episodio terminado
                episode_rewards.append(episode_raw_reward)
                episode_raw_reward = 0.0

                game.new_episode()
                state, frame_stack = init_frame_stack(game)

            # seleccionar acción
            action_idx, log_prob, entropy, value = get_action_and_value(net, state)
            action = actions[action_idx]

            # ejecutar acción
            reward_env = game.make_action(action)
            done = game.is_episode_finished()

            episode_raw_reward += reward_env

            # escalar recompensa para estabilidad
            reward = reward_env * REWARD_SCALE

            # guardar en buffer
            states.append(state.copy())
            action_idxs.append(action_idx)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())
            entropies.append(entropy.item())

            steps_collected += 1
            global_step += 1

            if not done:
                state, frame_stack = update_frame_stack(frame_stack, game)

        # bootstrap last value
        if game.is_episode_finished():
            last_value = 0.0
        else:
            _, _, _, value = get_action_and_value(net, state)
            last_value = value.item()

        # numpy
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.bool_)
        values_np = np.array(values, dtype=np.float32)

        advantages_np, returns_np = compute_gae(
            rewards_np, dones_np, values_np, last_value, GAMMA, LAMBDA
        )

        # normalizar advantages
        adv_mean = advantages_np.mean()
        adv_std = advantages_np.std() + 1e-8
        advantages_np = (advantages_np - adv_mean) / adv_std

        # tensores para update PPO
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(action_idxs, dtype=torch.long, device=DEVICE)
        old_log_probs_t = torch.tensor(log_probs, dtype=torch.float32, device=DEVICE)
        returns_t = torch.tensor(returns_np, dtype=torch.float32, device=DEVICE)
        advantages_t = torch.tensor(advantages_np, dtype=torch.float32, device=DEVICE)

        # update PPO (also returns avg grad norm so we can watch descent magnitude)
        avg_grad_norm = ppo_update(
            net, optimizer, states_t, actions_t,
            old_log_probs_t, returns_t, advantages_t
        )

        # logging
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards[-10:])
        else:
            mean_reward = 0.0

        # TensorBoard logs
        writer.add_scalar("reward/mean_last_10", mean_reward, update)
        writer.add_scalar("reward/episode_count", len(episode_rewards), update)
        writer.add_scalar("train/avg_grad_norm", avg_grad_norm, update)
        writer.add_scalar("train/steps_collected", global_step, update)

        print(
            f"Update {update}/{NUM_UPDATES} | "
            f"Episodes: {len(episode_rewards)} | "
            f"MeanReward(últ.10): {mean_reward:.2f} | "
            f"GradNorm: {avg_grad_norm:.3f}"
        )

    game.close()
    writer.close()
    print("Entrenamiento terminado.")
    plot_rewards(episode_rewards)
    torch.save(net.state_dict(), "vizdoom_ppo_basic.pth")
    print("Modelo guardado en vizdoom_ppo_basic.pth")


def plot_rewards(episode_rewards):
    plt.figure(figsize=(8,4))
    plt.plot(episode_rewards, label="Reward")
    if len(episode_rewards) > 10:
        window = max(5, len(episode_rewards)//20)
        kernel = np.ones(window) / window
        smoothed = np.convolve(episode_rewards, kernel, mode="valid")
        plt.plot(range(window-1, window-1+len(smoothed)), smoothed, label="Smoothed")
    plt.xlabel("Episodio")
    plt.ylabel("Reward total (no escalada)")
    plt.title("PPO en ViZDoom (basic.cfg)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_ppo()
