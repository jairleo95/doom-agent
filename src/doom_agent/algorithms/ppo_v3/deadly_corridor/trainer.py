import os

import numpy as np
import torch
import torch.optim as optim
from torch import amp
from torch.utils.tensorboard import SummaryWriter

from .config import (
    PPO_CFG,
    CKPT_CFG,
    ENV_CFG,
    CURRICULUM_STAGES,
    DEVICE,
    SHAPING_CFG,
    RUNS_DIR,
)
from .algo import compute_gae, ppo_update
from .curriculum import CurriculumManager
from .envs import ParallelEnv, get_deadly_corridor_actions
from .model import ActorCritic, get_value


class PPOTrainer:
    def __init__(
        self,
        stages=CURRICULUM_STAGES,
        ppo_cfg=PPO_CFG,
        ckpt_cfg=CKPT_CFG,
        env_cfg=ENV_CFG,
        shaping_cfg=SHAPING_CFG,
        log_dir=str(RUNS_DIR),
    ):
        self.stages = stages
        self.ppo = ppo_cfg
        self.ckpt = ckpt_cfg
        self.env_cfg = env_cfg
        self.shaping_cfg = shaping_cfg
        self.log_dir = log_dir

    def _entropy_coef(self, update):
        schedule_end = max(1, min(self.ppo.num_updates, 300))
        t = min(1.0, update / schedule_end)
        return self.ppo.entropy_start + t * (self.ppo.entropy_end - self.ppo.entropy_start)

    def _save_checkpoint(self, net, update, best=False):
        os.makedirs(self.ckpt.directory, exist_ok=True)
        if best:
            path = os.path.join(self.ckpt.directory, self.ckpt.best_name)
        elif isinstance(update, int):
            path = os.path.join(self.ckpt.directory, self.ckpt.periodic_template.format(update))
        else:
            path = os.path.join(self.ckpt.directory, "final.pth")
        torch.save(net.state_dict(), path)
        return path

    def train(self):
        curriculum = CurriculumManager(list(self.stages))
        current_stage = curriculum.current_stage
        vec_env = ParallelEnv(self.env_cfg.n_envs, current_stage, self.shaping_cfg)
        env_states = vec_env.reset()
        actions = get_deadly_corridor_actions()
        n_actions = len(actions)

        net = ActorCritic(self.env_cfg.frame_stack, n_actions).to(DEVICE)
        optimizer = optim.Adam(net.parameters(), lr=self.ppo.lr)
        writer = SummaryWriter(log_dir=self.log_dir)
        scaler = amp.GradScaler("cuda", enabled=self.ppo.use_amp and DEVICE.type == "cuda")

        episode_rewards = []
        best_mean_reward = float("-inf")

        print(f"[Curriculum] Etapa inicial -> {curriculum.describe_stage()}")

        for update in range(1, self.ppo.num_updates + 1):
            rollout_states = []
            action_idxs = []
            rewards = []
            dones = []
            log_probs = []
            values = []

            for _ in range(self.ppo.rollout_steps):
                state_batch = torch.tensor(np.array(env_states), dtype=torch.float32, device=DEVICE)
                with amp.autocast("cuda", enabled=scaler.is_enabled()):
                    logits_batch, values_batch = net(state_batch)
                probs_batch = torch.softmax(logits_batch.float(), dim=-1)
                dist_batch = torch.distributions.Categorical(probs_batch)
                action_batch = dist_batch.sample()
                log_prob_batch = dist_batch.log_prob(action_batch)

                actions_np = action_batch.cpu().numpy()
                next_states, step_rewards, step_dones, infos = vec_env.step(actions_np)

                for env_idx in range(self.env_cfg.n_envs):
                    rollout_states.append(env_states[env_idx].copy())
                    action_idxs.append(int(actions_np[env_idx]))
                    rewards.append(float(step_rewards[env_idx]))
                    dones.append(bool(step_dones[env_idx]))
                    log_probs.append(float(log_prob_batch[env_idx].item()))
                    values.append(float(values_batch[env_idx].item()))

                    if "episode_reward" in infos[env_idx]:
                        episode_rewards.append(infos[env_idx]["episode_reward"])

                env_states = next_states

            last_values = [get_value(net, env_states[i]) for i in range(self.env_cfg.n_envs)]

            rewards_np = np.array(rewards, dtype=np.float32)
            dones_np = np.array(dones, dtype=np.bool_)
            values_np = np.array(values, dtype=np.float32)

            total_steps = len(rewards_np)
            advantages_np = np.zeros_like(rewards_np, dtype=np.float32)
            returns_np = np.zeros_like(rewards_np, dtype=np.float32)

            for env_idx in range(self.env_cfg.n_envs):
                env_slice = slice(env_idx, total_steps, self.env_cfg.n_envs)
                adv_env, ret_env = compute_gae(
                    rewards_np[env_slice],
                    dones_np[env_slice],
                    values_np[env_slice],
                    last_values[env_idx],
                    self.ppo.gamma,
                    self.ppo.lam,
                )
                advantages_np[env_slice] = adv_env
                returns_np[env_slice] = ret_env

            returns_np = (returns_np - returns_np.mean()) / (returns_np.std() + 1e-8)

            adv_mean = advantages_np.mean()
            adv_std = advantages_np.std() + 1e-8
            advantages_np = (advantages_np - adv_mean) / adv_std

            states_t = torch.tensor(np.array(rollout_states), dtype=torch.float32, device=DEVICE)
            actions_t = torch.tensor(action_idxs, dtype=torch.long, device=DEVICE)
            old_log_probs_t = torch.tensor(log_probs, dtype=torch.float32, device=DEVICE)
            returns_t = torch.tensor(returns_np, dtype=torch.float32, device=DEVICE)
            advantages_t = torch.tensor(advantages_np, dtype=torch.float32, device=DEVICE)

            ent_coef = self._entropy_coef(update)
            writer.add_scalar("ppo/entropy_coef", ent_coef, update)

            stats = ppo_update(
                net,
                optimizer,
                states_t,
                actions_t,
                old_log_probs_t,
                returns_t,
                advantages_t,
                ent_coef,
                self.ppo,
                scaler=scaler,
            )

            mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            if mean_reward > best_mean_reward and episode_rewards:
                best_mean_reward = mean_reward
                best_path = self._save_checkpoint(net, update=None, best=True)
                writer.add_scalar("train/best_mean_reward", best_mean_reward, update)
                print(f"[Checkpoint] Nuevo mejor modelo guardado en {best_path} (mean_reward={best_mean_reward:.2f})")

            writer.add_scalar("ppo/kl", stats["approx_kl"], update)
            writer.add_scalar("ppo/clipfrac", stats["clipfrac"], update)
            writer.add_scalar("ppo/entropy", stats["entropy"], update)
            writer.add_scalar("loss/value", stats["value_loss"], update)
            writer.add_scalar("loss/policy", stats["policy_loss"], update)
            writer.add_scalar("train/avg_grad_norm", stats["grad_norm"], update)

            print(
                f"[DeadlyCorridor] Update {update}/{self.ppo.num_updates} | "
                f"Episodes: {len(episode_rewards)} | "
                f"MeanReward(últ.10): {mean_reward:.2f} | "
                f"KL:{stats['approx_kl']:.4f} Clip:{stats['clipfrac']:.2f} |"
                f"H:{stats['entropy']:.3f} VLoss:{stats['value_loss']:.2f}"
            )

            if update % self.ckpt.interval == 0:
                periodic_path = self._save_checkpoint(net, update)
                writer.add_scalar("checkpoint/last_interval", update, update)
                print(f"[Checkpoint] Guardado periódico en {periodic_path}")

            if curriculum.maybe_advance(episode_rewards):
                current_stage = curriculum.current_stage
                env_states = vec_env.set_stage(current_stage)
                writer.add_scalar("curriculum/stage_index", curriculum.current_idx, update)
                writer.add_text("curriculum/stage_name", current_stage.name, update)
                print(f"[Curriculum] Avanzando a etapa -> {curriculum.describe_stage()}")

        vec_env.close()
        writer.close()
        print("Entrenamiento terminado.")
        final_path = self._save_checkpoint(net, update=None, best=False)
        print(f"Modelo final guardado en {final_path}")


def plot_rewards(episode_rewards):
    if len(episode_rewards) == 0:
        print("No hay episodios registrados.")
        return

    import matplotlib.pyplot as plt  # local import to avoid UI issues when not plotting

    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, label="Reward")
    if len(episode_rewards) > 10:
        window = max(5, len(episode_rewards) // 20)
        kernel = np.ones(window) / window
        smoothed = np.convolve(episode_rewards, kernel, mode="valid")
        plt.plot(range(window - 1, window - 1 + len(smoothed)), smoothed, label="Smoothed")
    plt.xlabel("Episodio")
    plt.ylabel("Reward total (no escalada)")
    plt.title("PPO en ViZDoom - deadly_corridor")
    plt.legend()
    plt.tight_layout()
    plt.show()
