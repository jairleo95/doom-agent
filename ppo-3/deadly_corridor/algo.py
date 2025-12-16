import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from .config import PPOConfig


def compute_gae(rewards, dones, values, last_value, gamma, lam):
    T = len(rewards)
    values = np.append(values, last_value)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


def ppo_update(
    net,
    optimizer,
    states,
    actions,
    old_log_probs,
    returns,
    advantages,
    entropy_coef,
    cfg: PPOConfig,
    scaler: GradScaler | None = None,
):
    num_steps = states.size(0)
    indices = np.arange(num_steps)

    grad_norms, kls, clipfracs, entropies, vlosses, plosses = ([] for _ in range(6))

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)

        for start in range(0, num_steps, cfg.mini_batch_size):
            end = start + cfg.mini_batch_size
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
            clipfrac = (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean()

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (mb_returns - values).pow(2).mean()

            loss = policy_loss + cfg.value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                optimizer.step()

            grad_norms.append(float(grad_norm.item()))
            kls.append(float(approx_kl.item()))
            clipfracs.append(float(clipfrac.item()))
            entropies.append(float(entropy.item()))
            vlosses.append(float(value_loss.item()))
            plosses.append(float(policy_loss.item()))

        if len(kls) > 0 and np.mean(kls[-max(1, num_steps // cfg.mini_batch_size):]) > cfg.target_kl:
            break

    return {
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "approx_kl": float(np.mean(kls)) if kls else 0.0,
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "value_loss": float(np.mean(vlosses)) if vlosses else 0.0,
        "policy_loss": float(np.mean(plosses)) if plosses else 0.0,
    }
