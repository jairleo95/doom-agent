import random
from collections import deque

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

# ==========================
#  CONFIG GENERAL
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPISODES = 1000          # súbelo a 1000+ para algo más sólido
GAMMA = 0.99
LAMBDA = 0.95               # para GAE
LR = 5e-5          # antes 1e-4
ENTROPY_BETA = 0.02  # antes 0.01
         # peso de entropía (exploración)
VALUE_LOSS_COEF = 0.5       # peso del loss de valor
MAX_STEPS_PER_EPISODE = 400
FRAME_STACK = 4             # número de frames apilados

EPS_START = 0.3
EPS_END = 0.02
EPS_DECAY_EPISODES = 600


# ==========================
#  VIZDOOM HELPERS
# ==========================

def create_doom_game(config_path="basic.cfg"):
    game = DoomGame()
    game.load_config(config_path)
    game.set_window_visible(True)  # False para Colab / headless
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
    Salida: (1,84,84)
    """
    if frame.ndim == 3 and frame.shape[0] <= 4:
        frame = np.transpose(frame, (1, 2, 0))  # (H,W,C)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[np.newaxis, :, :]  # (1,H,W)


def init_frame_stack(game):
    """
    Inicializa el frame stack con FRAME_STACK copias del primer frame.
    Devuelve np.array con shape (FRAME_STACK,84,84)
    """
    state = game.get_state()
    frame = preprocess_frame(state.screen_buffer)  # (1,84,84)
    stack = deque([frame for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
    return stack_to_state(stack), stack


def stack_to_state(stack):
    """
    Convierte deque de frames (FRAME_STACK,1,84,84) -> (FRAME_STACK,84,84)
    """
    return np.concatenate(list(stack), axis=0)


def update_frame_stack(stack, game):
    """
    Mete frame nuevo al stack y devuelve el nuevo estado (FRAME_STACK,84,84)
    """
    frame = preprocess_frame(game.get_state().screen_buffer)  # (1,84,84)
    stack.append(frame)
    return stack_to_state(stack), stack


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

ALPHA_EXPL = 0.2  # mezcla con uniforme: 0 = nada, 0.2 = 20% uniforme


def select_action(net, state, n_actions):
    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    logits, value = net(state_t)  # logits: (1,n_actions), value: (1,1)

    # evitamos logits demasiado grandes (más estabilidad)
    logits = torch.clamp(logits, -10.0, 10.0)

    probs = torch.softmax(logits, dim=-1)

    # --- mezcla con uniforme para evitar colapso ---
    # probs' = (1-α)*probs + α * 1/n
    if ALPHA_EXPL > 0:
        uniform = torch.ones_like(probs) / n_actions
        probs = (1.0 - ALPHA_EXPL) * probs + ALPHA_EXPL * uniform

    dist = torch.distributions.Categorical(probs)

    action_idx = dist.sample()
    log_prob = dist.log_prob(action_idx)
    entropy = dist.entropy()

    value = value.squeeze()  # escalar ()

    return (
        int(action_idx.item()),
        log_prob.squeeze(0),
        value,
        entropy.squeeze(0),
    )



def compute_gae(rewards, values, dones, gamma, lam):
    """
    Calcula GAE y returns.
    rewards: list[T]
    values:  list[T+1]  (incluye V(s_{T}) bootstrap)
    dones:   list[T] (bool)
    Devuelve:
      advantages: np.array(T,)
      returns:    np.array(T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


# ==========================
#  ENTRENAMIENTO A2C + GAE
# ==========================

def train_a2c(game, net, optimizer, actions, num_episodes, writer=None):
    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        game.new_episode()

        # inicializar frame stack
        state, frame_stack = init_frame_stack(game)

        log_probs = []
        values = []
        rewards = []
        entropies = []
        dones = []

        total_reward = 0.0
        epsilon = max(
            EPS_END,
            EPS_START - (EPS_START - EPS_END) * (ep / EPS_DECAY_EPISODES)
        )

        for step in range(MAX_STEPS_PER_EPISODE):
            # si ya terminó el episodio, salimos
            if game.is_episode_finished():
                break

            # seleccionar acción con la política
            # epsilon scheduling
            if random.random() < epsilon:
                # acción totalmente aleatoria
                action_idx = random.randrange(len(actions))

                # igual calculamos log_prob/value/entropy para esa acción
                state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits, value = net(state_t)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                # log_prob de esa acción concreta
                action_idx_t = torch.tensor([action_idx], device=DEVICE)
                log_prob = dist.log_prob(action_idx_t)[0]
                entropy = dist.entropy()[0]
                value = value.squeeze()
            else:
                action_idx, log_prob, value, entropy = select_action(net, state, len(actions))

            action = actions[action_idx]

            # ejecutar acción
            reward = game.make_action(action)
            # Escalar recompensas
            reward = reward / 25.0
            total_reward += reward

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

            # revisar si terminó DESPUÉS de actuar
            done = game.is_episode_finished()
            dones.append(done)

            if done:
                break

            # actualizar estado (frame stack) con el próximo frame
            state, frame_stack = update_frame_stack(frame_stack, game)

        # ------ bootstrap V(s_T) ------
        if len(rewards) == 0:
            # episodio vacío (edge case raro)
            episode_rewards.append(0.0)
            if writer is not None:
                writer.add_scalar("reward/episode_total", 0.0, ep)
                writer.add_scalar("episode/length", 0, ep)
                writer.add_scalar("policy/epsilon", epsilon, ep)
            continue

        if dones[-1]:
            # terminal real → V(s_T) = 0
            next_value = torch.tensor(0.0, device=DEVICE)
        else:
            # cortado por MAX_STEPS → bootstrap con V(s_T)
            if not game.is_episode_finished():
                # tomar el estado actual del juego
                state, frame_stack = update_frame_stack(frame_stack, game)
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            _, next_value = net(state_t)
            next_value = next_value.squeeze()  # escalar

        episode_rewards.append(total_reward)

        # ------ GAE + returns ------
        values_t = torch.stack(values)  # (T,) porque value es escalar
        # concatenamos el valor bootstrap al final -> (T+1,)
        values_all = torch.cat(
            [values_t, next_value.unsqueeze(0)], dim=0
        ).detach().cpu().numpy()

        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.bool_)

        advantages_np, returns_np = compute_gae(
            rewards_np, values_all, dones_np, GAMMA, LAMBDA
        )

        # normalizar ventajas
        adv_mean = advantages_np.mean()
        adv_std = advantages_np.std() + 1e-8
        advantages_np = (advantages_np - adv_mean) / adv_std

        returns_t = torch.tensor(returns_np, dtype=torch.float32, device=DEVICE)
        advantages_t = torch.tensor(advantages_np, dtype=torch.float32, device=DEVICE)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        # losses
        value_loss = (returns_t - values_t).pow(2).mean()
        policy_loss = -(log_probs_t * advantages_t).mean()
        entropy_loss = -entropies_t.mean()

        loss = policy_loss + VALUE_LOSS_COEF * value_loss + ENTROPY_BETA * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer.step()

        mean_entropy = entropies_t.mean().item()
        print(
            f"Ep {ep}/{num_episodes} | "
            f"Reward: {total_reward:.2f} | "
            f"Len: {len(rewards)} | "
            f"Loss: {loss.item():.4f} | "
            f"H(ent): {mean_entropy:.3f} | eps: {epsilon:.3f}"
        )

        if writer is not None:
            writer.add_scalar("reward/episode_total", total_reward, ep)
            writer.add_scalar("episode/length", len(rewards), ep)
            writer.add_scalar("policy/epsilon", epsilon, ep)
            writer.add_scalar("loss/total", loss.item(), ep)
            writer.add_scalar("loss/policy", policy_loss.item(), ep)
            writer.add_scalar("loss/value", value_loss.item(), ep)
            writer.add_scalar("loss/entropy", entropy_loss.item(), ep)
            writer.add_scalar("entropy/mean", mean_entropy, ep)

    return episode_rewards



def plot_rewards(episode_rewards):
    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, label="Reward")
    window = max(1, len(episode_rewards) // 20)
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(episode_rewards, kernel, mode="valid")
        plt.plot(range(window - 1, window - 1 + len(smoothed)), smoothed, label="Smoothed")
    plt.xlabel("Episodio")
    plt.ylabel("Reward total")
    plt.title("Curva de aprendizaje A2C+GAE en ViZDoom (basic.cfg)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================
#  MAIN
# ==========================

def main():
    game = create_doom_game("basic.cfg")
    actions = get_basic_actions()

    n_actions = len(actions)
    net = A2CNet(input_channels=FRAME_STACK, n_actions=n_actions).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    writer = SummaryWriter(log_dir="runs/vizdoom_a2c_gae_basic")

    rewards = train_a2c(game, net, optimizer, actions, NUM_EPISODES, writer=writer)
    game.close()
    writer.close()

    print("Entrenamiento terminado.")
    plot_rewards(rewards)

    # guardar modelo
    torch.save(net.state_dict(), "vizdoom_a2c_gae_basic.pth")
    print("Modelo guardado en vizdoom_a2c_gae_basic.pth")


if __name__ == "__main__":
    main()
