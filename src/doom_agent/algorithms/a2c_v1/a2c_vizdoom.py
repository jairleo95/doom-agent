import random
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from doom_agent.paths import scenario_path
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

# ==========================
#  CONFIG GENERAL
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 300          # súbelo a 1000+ para algo decente
GAMMA = 0.99
LR = 1e-4
ENTROPY_BETA = 0.01         # peso de la entropía (exploración)
VALUE_LOSS_COEF = 0.5       # peso del crítico
MAX_STEPS_PER_EPISODE = 400


# ==========================
#  VIZDOOM HELPERS
# ==========================

def create_doom_game(config_path="basic.cfg"):
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(scenario_path(config_path))

    game = DoomGame()
    game.load_config(str(config_file))
    game.set_window_visible(True)  # ponlo en False si no quieres la ventana
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.init()
    return game


def get_basic_actions():
    # Para basic.cfg: izquierda, derecha, disparar
    return [
        [1, 0, 0],  # left
        [0, 1, 0],  # right
        [0, 0, 1],  # shoot
    ]


def preprocess_frame(frame, new_size=(84, 84)):
    """
    Convierte frame de ViZDoom a escala de grises 84x84 normalizado [0,1].
    Entrada frame: (C,H,W) con C=3.
    """
    if frame.ndim == 3 and frame.shape[0] <= 4:
        frame = np.transpose(frame, (1, 2, 0))  # (H,W,C)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    # añadimos canal (1,84,84)
    return normalized[np.newaxis, :, :]


# ==========================
#  RED A2C
# ==========================

class A2CNet(nn.Module):
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

        # 84x84 -> 20x20 -> 9x9 -> 7x7 => 64*7*7 = 3136
        self.fc = nn.Linear(64 * 7 * 7, 512)

        # Cabezas
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))

        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


# ==========================
#  FUNCIONES RL
# ==========================

def compute_returns(rewards, dones, gamma):
    """
    Calcula returns G_t con episodios posiblemente truncados.
    rewards: [T]
    dones:   [T] (bool)
    """
    G = 0.0
    returns = []

    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0.0
        G = r + gamma * G
        returns.append(G)

    returns.reverse()
    return np.array(returns, dtype=np.float32)


def select_action(net, state, n_actions):
    """
    state: np.array (1, 84, 84)
    """
    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,1,84,84)
    logits, value = net(state_t)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)

    action_idx = dist.sample()
    log_prob = dist.log_prob(action_idx)
    entropy = dist.entropy()

    return (
        int(action_idx.item()),
        log_prob.squeeze(0),
        value.squeeze(0),
        entropy.squeeze(0),
    )


# ==========================
#  ENTRENAMIENTO A2C
# ==========================

def train_a2c(game, net, optimizer, actions, num_episodes):
    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        game.new_episode()
        state = preprocess_frame(game.get_state().screen_buffer)

        log_probs = []
        values = []
        rewards = []
        entropies = []
        dones = []

        total_reward = 0.0

        for step in range(MAX_STEPS_PER_EPISODE):
            if game.is_episode_finished():
                done = True
            else:
                done = False

            if done:
                dones.append(True)
                break

            action_idx, log_prob, value, entropy = select_action(net, state, len(actions))
            action = actions[action_idx]

            reward = game.make_action(action)
            total_reward += reward

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            dones.append(False)

            if game.is_episode_finished():
                break

            next_state = preprocess_frame(game.get_state().screen_buffer)
            state = next_state

        # Si el episodio terminó por MAX_STEPS_PER_EPISODE y no game.is_episode_finished(),
        # podríamos bootstrapear con V(s_T), pero para simplificar lo tratamos como done.

        episode_rewards.append(total_reward)

        # --- Calcular returns y advantages ---
        if len(rewards) == 0:
            # episodio vacío (muy raro, pero por seguridad)
            continue

        returns = compute_returns(rewards, dones, GAMMA)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        values_t = torch.stack(values)  # (T,)
        log_probs_t = torch.stack(log_probs)  # (T,)
        entropies_t = torch.stack(entropies)  # (T,)

        advantages = returns_t - values_t.detach()

        # --- Losses ---
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss = (returns_t - values_t).pow(2).mean()
        entropy_loss = -entropies_t.mean()

        loss = policy_loss + VALUE_LOSS_COEF * value_loss + ENTROPY_BETA * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

        print(
            f"Ep {ep}/{num_episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Len: {len(rewards)} | "
            f"Loss: {loss.item():.4f}"
        )

    return episode_rewards


def plot_rewards(episode_rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards)
    window = max(1, len(episode_rewards) // 20)
    if window > 1:
        # media móvil sencilla
        kernel = np.ones(window) / window
        smoothed = np.convolve(episode_rewards, kernel, mode="valid")
        plt.plot(range(window - 1, window - 1 + len(smoothed)), smoothed)
    plt.xlabel("Episodio")
    plt.ylabel("Reward total")
    plt.title("Curva de aprendizaje A2C en ViZDoom (basic.cfg)")
    plt.legend(["Reward", "Reward suavizado"] if window > 1 else ["Reward"])
    plt.tight_layout()
    plt.show()


# ==========================
#  MAIN
# ==========================

def main():
    game = create_doom_game("basic.cfg")
    actions = get_basic_actions()

    n_actions = len(actions)
    net = A2CNet(input_channels=1, n_actions=n_actions).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)

    rewards = train_a2c(game, net, optimizer, actions, NUM_EPISODES)
    game.close()

    print("Entrenamiento terminado.")
    plot_rewards(rewards)

    # guardar modelo
    torch.save(net.state_dict(), "vizdoom_a2c_basic.pth")
    print("Modelo guardado en vizdoom_a2c_basic.pth")


if __name__ == "__main__":
    main()
