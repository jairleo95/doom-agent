import os
import random
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution

from doom_agent.paths import scenario_path

# =========================================
# SEED (opcional pero recomendable)
# =========================================
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================================
# CONFIG
# =========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "deadly_corridor"
LOG_DIR = BASE_DIR / "runs" / "vizdoom_ppo_deadly_corridor"

# RL
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.1
LR = 1e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

# PPO
ROLLOUT_STEPS = 4096       # deadly_corridor es más difícil, usamos más pasos
NUM_UPDATES = 600          # sube/baja según paciencia
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 256
CHECKPOINT_INTERVAL = 25
BEST_MODEL_NAME = "best.pth"
PERIODIC_MODEL_TEMPLATE = "update_{:04d}.pth"

# Observaciones
FRAME_STACK = 4
IMG_SIZE = (84, 84)

CURRICULUM_STAGES = [
    {
        "name": "skill2_warmup",
        "doom_skill": 2,
        "living_reward": -0.01,
        "reward_scale": 1.0 / 30.0,
        "min_episodes": 6,
        "unlock_mean_reward": -15.0,
        "window": 4,
    },
    {
        "name": "skill4_standard",
        "doom_skill": 4,
        "living_reward": 0.0,
        "reward_scale": 1.0 / 40.0,
        "min_episodes": 8,
        "unlock_mean_reward": 40.0,
        "window": 6,
    },
    {
        "name": "skill5_target",
        "doom_skill": 5,
        "living_reward": 0.0,
        "reward_scale": 1.0 / 50.0,
        "min_episodes": 12,
        "unlock_mean_reward": 75.0,
        "window": 8,
    },
]

# =========================================
# ENV HELPERS (ViZDoom deadly_corridor)
# =========================================

def create_doom_game(
    config_path="deadly_corridor.cfg",
    visible=False,
    doom_skill=5,
    living_reward=0.0,
):
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(scenario_path(config_path))

    game = DoomGame()
    game.load_config(str(config_file))
    if doom_skill is not None:
        game.set_doom_skill(int(doom_skill))
    if living_reward is not None:
        game.set_living_reward(float(living_reward))
    game.set_window_visible(visible)  # pon True si quieres ver la ventana
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.init()
    return game


def get_deadly_corridor_actions():
    # Orden de botones asumido:
    # [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT]
    return [
        # 0: forward
        [0,0,0,1,0,0,0],
        # 1: attack
        [0,0,1,0,0,0,0],
        # 2: forward + attack
        [0,0,1,1,0,0,0],

        # 3: turn left
        [0,0,0,0,0,1,0],
        # 4: turn right
        [0,0,0,0,0,0,1],

        # 5: turn left + attack
        [0,0,1,0,0,1,0],
        # 6: turn right + attack
        [0,0,1,0,0,0,1],

        # 7: strafe left
        [1,0,0,0,0,0,0],
        # 8: strafe right
        [0,1,0,0,0,0,0],

        # 9: backward
        [0,0,0,0,1,0,0],

        # 10: forward + turn left (micro-ajuste de mira)
        [0,0,0,1,0,1,0],
        # 11: forward + turn right
        [0,0,0,1,0,0,1],
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
    return normalized[np.newaxis, :, :]  # (1,H,W)


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

TARGET_KL = 0.02  # puedes ajustar 0.01–0.03

def ppo_update(net, optimizer, states, actions, old_log_probs, returns, advantages):
    num_steps = states.size(0)
    indices = np.arange(num_steps)

    grad_norms = []
    kls = []
    clipfracs = []
    entropies = []
    vlosses = []
    plosses = []

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

            log_ratio = new_log_probs - mb_old_log_probs
            ratio = torch.exp(log_ratio)

            approx_kl = (mb_old_log_probs - new_log_probs).mean()
            clipfrac = (torch.abs(ratio - 1.0) > CLIP_EPS).float().mean()

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (mb_returns - values).pow(2).mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            grad_norms.append(float(grad_norm.item()))
            kls.append(float(approx_kl.item()))
            clipfracs.append(float(clipfrac.item()))
            entropies.append(float(entropy.item()))
            vlosses.append(float(value_loss.item()))
            plosses.append(float(policy_loss.item()))

        # early stop PPO si KL se dispara
        if len(kls) > 0 and np.mean(kls[-max(1, num_steps // MINI_BATCH_SIZE):]) > TARGET_KL:
            break

    return {
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "value_loss": float(np.mean(vlosses)) if vlosses else 0.0,
        "policy_loss": float(np.mean(plosses)) if plosses else 0.0,
    }


# =========================================
# TRAIN LOOP PPO PARA DEADLY_CORRIDOR
# =========================================

class CurriculumManager:
    def __init__(self, stages):
        self.stages = stages
        self.current_idx = 0
        self.stage_start_episode = 0

    @property
    def current_stage(self):
        return self.stages[self.current_idx]

    def maybe_advance(self, episode_rewards):
        if self.current_idx >= len(self.stages) - 1:
            return False

        stage = self.current_stage
        total_eps = len(episode_rewards)
        stage_eps = total_eps - self.stage_start_episode

        if stage_eps < stage["min_episodes"]:
            return False

        window = min(stage.get("window", stage["min_episodes"]), stage_eps, len(episode_rewards))
        if window <= 0:
            return False

        mean_reward = float(np.mean(episode_rewards[-window:]))
        if mean_reward >= stage["unlock_mean_reward"]:
            self.current_idx += 1
            self.stage_start_episode = total_eps
            return True
        return False

    def describe_stage(self):
        stage = self.current_stage
        return (
            f"{self.current_idx + 1}/{len(self.stages)} "
            f"{stage['name']} (skill={stage['doom_skill']}, "
            f"living_reward={stage['living_reward']}, "
            f"reward_scale={stage['reward_scale']})"
        )

@torch.no_grad()
def get_value(net, state):
    state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    _, value = net(state_t)
    return float(value.squeeze(0).item())

ENTROPY_START = 0.02
ENTROPY_END = 0.005

def entropy_coef(update):
    # baja en ~300 updates
    t = min(1.0, update / 300.0)
    return ENTROPY_START + t * (ENTROPY_END - ENTROPY_START)


def train_ppo_deadly_corridor():
    curriculum = CurriculumManager(CURRICULUM_STAGES)
    current_stage = curriculum.current_stage
    reward_scale = current_stage["reward_scale"]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def build_game_for_stage(stage):
        g = create_doom_game(
            "deadly_corridor.cfg",
            visible=True,
            doom_skill=stage["doom_skill"],
            living_reward=stage["living_reward"],
        )
        g.new_episode()
        s, stack = init_frame_stack(g)
        return g, s, stack

    game, state, frame_stack = build_game_for_stage(current_stage)
    actions = get_deadly_corridor_actions()
    n_actions = len(actions)

    net = ActorCritic(FRAME_STACK, n_actions).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    writer = SummaryWriter(log_dir=str(LOG_DIR))

    episode_rewards = []
    episode_raw_reward = 0.0
    global_step = 0
    best_mean_reward = float("-inf")

    print(f"[Curriculum] Etapa inicial -> {curriculum.describe_stage()}")

    for update in range(1, NUM_UPDATES + 1):
        # buffers
        states = []
        action_idxs = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        steps_collected = 0

        while steps_collected < ROLLOUT_STEPS:
            if game.is_episode_finished():
                episode_rewards.append(episode_raw_reward)
                episode_raw_reward = 0.0

                game.new_episode()
                state, frame_stack = init_frame_stack(game)

            action_idx, log_prob, _, value = get_action_and_value(net, state)
            action = actions[action_idx]

            reward_env = game.make_action(action)
            done = game.is_episode_finished()

            episode_raw_reward += reward_env

            # reward escalado
            reward = reward_env * reward_scale

            states.append(state.copy())
            action_idxs.append(action_idx)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())

            steps_collected += 1
            global_step += 1

            if not done:
                state, frame_stack = update_frame_stack(frame_stack, game)

        # bootstrap last value
        if game.is_episode_finished():
            last_value = 0.0
        else:
            last_value = 0.0 if game.is_episode_finished() else get_value(net, state)

        # numpy
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.bool_)
        values_np = np.array(values, dtype=np.float32)

        advantages_np, returns_np = compute_gae(
            rewards_np, dones_np, values_np, last_value, GAMMA, LAMBDA
        )
        returns_np = (returns_np - returns_np.mean()) / (returns_np.std() + 1e-8)


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

        ENTROPY_COEF = entropy_coef(update)
        writer.add_scalar("ppo/entropy_coef", ENTROPY_COEF, update)

        # update PPO (retorna grad_norm promedio para monitorear)
        stats = ppo_update(
            net, optimizer, states_t, actions_t,
            old_log_probs_t, returns_t, advantages_t
        )

        # logging
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards[-10:])
        else:
            mean_reward = 0.0
        if mean_reward > best_mean_reward and len(episode_rewards) > 0:
            best_mean_reward = mean_reward
            best_path = CHECKPOINT_DIR / BEST_MODEL_NAME
            torch.save(net.state_dict(), best_path)
            writer.add_scalar("train/best_mean_reward", best_mean_reward, update)
            print(f"[Checkpoint] Nuevo mejor modelo guardado en {best_path} (mean_reward={best_mean_reward:.2f})")

        writer.add_scalar("ppo/kl", stats["approx_kl"], update)
        writer.add_scalar("ppo/clipfrac", stats["clipfrac"], update)
        writer.add_scalar("ppo/entropy", stats["entropy"], update)
        writer.add_scalar("loss/value", stats["value_loss"], update)
        writer.add_scalar("loss/policy", stats["policy_loss"], update)
        writer.add_scalar("train/avg_grad_norm", stats["grad_norm"], update)

        print(
            f"[DeadlyCorridor] Update {update}/{NUM_UPDATES} | "
            f"Episodes: {len(episode_rewards)} | "
            f"MeanReward(últ.10): {mean_reward:.2f} | "
            f"KL:{stats['approx_kl']:.4f} Clip:{stats['clipfrac']:.2f} |"
            f"H:{stats['entropy']:.3f} VLoss:{stats['value_loss']:.2f}"
        )

        if update % CHECKPOINT_INTERVAL == 0:
            periodic_path = CHECKPOINT_DIR / PERIODIC_MODEL_TEMPLATE.format(update)
            torch.save(net.state_dict(), periodic_path)
            writer.add_scalar("checkpoint/last_interval", update, update)
            print(f"[Checkpoint] Guardado periódico en {periodic_path}")

        if curriculum.maybe_advance(episode_rewards):
            current_stage = curriculum.current_stage
            reward_scale = current_stage["reward_scale"]
            game.close()
            game, state, frame_stack = build_game_for_stage(current_stage)
            episode_raw_reward = 0.0
            writer.add_scalar("curriculum/stage_index", curriculum.current_idx, update)
            writer.add_text("curriculum/stage_name", current_stage["name"], update)
            print(f"[Curriculum] Avanzando a etapa -> {curriculum.describe_stage()}")

    game.close()
    writer.close()
    print("Entrenamiento terminado.")
    plot_rewards(episode_rewards)
    final_path = CHECKPOINT_DIR / "final.pth"
    torch.save(net.state_dict(), final_path)
    print(f"Modelo final guardado en {final_path}")


def plot_rewards(episode_rewards):
    if len(episode_rewards) == 0:
        print("No hay episodios registrados.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, label="Reward")
    if len(episode_rewards) > 10:
        window = max(5, len(episode_rewards)//20)
        kernel = np.ones(window) / window
        smoothed = np.convolve(episode_rewards, kernel, mode="valid")
        plt.plot(range(window-1, window-1+len(smoothed)), smoothed, label="Smoothed")
    plt.xlabel("Episodio")
    plt.ylabel("Reward total (no escalada)")
    plt.title("PPO en ViZDoom - deadly_corridor")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_ppo_deadly_corridor()
