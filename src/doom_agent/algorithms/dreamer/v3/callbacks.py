"""
Training callbacks for Dreamer V3.

Provides callbacks for video recording, checkpointing, evaluation, and metrics logging.
"""

import os
import json
import imageio
import numpy as np
import cv2
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None

from doom_agent.algorithms.dreamer.v3.doom_envs import DoomDreamerEnv


class VideoRecorderCallback:
    """
    Registra episodios en GIF, inspirado en la implementación de PPO v5.
    Se dispara por frecuencia de pasos (global_step) y no por episodio.
    """

    def __init__(
        self,
        eval_env,
        agent,
        save_path: str = "videos",
        name_prefix: str = "dreamer_v3",
        render_freq: int = 100_000,  # pasos globales entre grabaciones
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        fps: int = 35,
    ):
        """
        Args:
            eval_env: Environment to grab frames from.
            agent: DreamerV3Agent instance.
            save_path: Directory to save videos.
            name_prefix: Prefix for video filenames.
            render_freq: Steps between recordings (use 0/None to disable).
            n_eval_episodes: Episodes to record per trigger.
            deterministic: Whether to use deterministic actions.
            fps: Frames per second in the saved GIF.
        """
        self.eval_env = eval_env
        self.agent = agent
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.fps = fps

        os.makedirs(save_path, exist_ok=True)

    def should_record(self, global_step: int) -> bool:
        return self.render_freq and self.render_freq > 0 and global_step % self.render_freq == 0

    def _obs_to_frame(self, obs: np.ndarray) -> np.ndarray:
        # Use cached high-res render if available, otherwise fallback to observation
        if hasattr(self.eval_env, 'last_high_res_render') and self.eval_env.last_high_res_render is not None:
            return self.eval_env.last_high_res_render.copy()
            
        if obs is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
            
        frame = obs
        if frame.ndim == 4:  # batch dim
            frame = frame[0]
            
        # Ensure RGB format for imageio
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        return frame

    def record_video(self, suffix: str):
        frames = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            self.agent.reset_state()
            done = False

            while not done:
                frames.append(self._obs_to_frame(obs))
                action = self.agent.select_action(obs, eval_mode=True)
                obs, reward, done = self.eval_env.step(action)

        if frames:
            save_file = os.path.join(self.save_path, f"{self.name_prefix}{suffix}.gif")
            # use duration instead of fps to avoid DeprecationWarning (duration is in ms)
            imageio.mimsave(save_file, frames, duration=1000/self.fps)
            print(f"  Saved video: {save_file}")


class ImaginationVideoCallback:
    """Callback to record 'imagination' videos (world model predictions) to TensorBoard."""
    
    def __init__(self, agent, log_dir, render_freq=1000):
        self.agent = agent # DreamerV3Agent (adapter)
        self.log_dir = Path(log_dir)
        self.render_freq = render_freq
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def should_render(self, global_step):
        # Caller manages frequency via last_imag_log_step to handle n_envs > 1 skip
        return self.render_freq and self.render_freq > 0
        
    def record_imagination(self, global_step, batch):
        """Record model predictions for a given batch of data."""
        if not self.should_render(global_step):
            return
            
        # agent.agent is the actual Dreamer model
        with torch.no_grad():
            try:
                # video_pred returns (Truth, Prediction, Error) concatenated
                # Shape is (Batch, Time, H, W*3, C)
                video = self.agent.agent.video_pred(batch)
                
                # Convert to (T, C, H, W) for TensorBoard (takes one video at a time)
                # We take the first 1 video in the batch
                video_tb = video[0] # (T, H, W, C)
                video_tb = video_tb.permute(0, 3, 1, 2) # (T, C, H, W)
                
                # TensorBoard wants (N, T, C, H, W) where N is number of videos
                video_tb = video_tb.unsqueeze(0)
                
                self.writer.add_video("imagination/truth_pred_error", video_tb, global_step, fps=15)

                # W&B Visualization (if active)
                if wandb and wandb.run:
                    # wandb.Video takes (T, H, W, C) or (T, C, H, W)
                    # Our video is (B, T, H, W*3, C)
                    # We convert to (T, H, W*3, C) for W&B
                    video_wb = video[0].cpu().numpy()
                    wandb.log({
                        "imagination/truth_pred_error": wandb.Video(video_wb, fps=15, format="gif")
                    }, step=global_step)

            except Exception as e:
                print(f"Warning: Failed to log imagination video: {e}")


class CheckpointCallback:
    """Saves model checkpoints periodically."""
    
    def __init__(
        self,
        agent,
        save_path='checkpoints',
        name_prefix='dreamer_v3',
        save_freq=100  # Save every N episodes
    ):
        """
        Args:
            agent: DreamerV3Agent instance
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint filenames
            save_freq: Save every N episodes
        """
        self.agent = agent
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_freq = save_freq
        
        os.makedirs(save_path, exist_ok=True)
    
    def should_save(self, episode):
        """Check if should save checkpoint."""
        return self.save_freq > 0 and episode % self.save_freq == 0
    
    def save_checkpoint(self, episode):
        """Save model checkpoint."""
        save_file = os.path.join(
            self.save_path,
            f"{self.name_prefix}_ep{episode}.pt"
        )
        self.agent.save(save_file)
        print(f"  Saved checkpoint: {save_file}")


class EvalCallback:
    """Evaluates agent performance periodically."""
    
    def __init__(
        self,
        eval_env,
        agent,
        eval_freq=50,  # Evaluate every N episodes
        n_eval_episodes=5,
        callback_on_new_best=None
    ):
        """
        Args:
            eval_env: Environment for evaluation
            agent: DreamerV3Agent instance
            eval_freq: Evaluate every N episodes
            n_eval_episodes: Number of episodes to evaluate
            callback_on_new_best: Callback to trigger on new best model
        """
        self.eval_env = eval_env
        self.agent = agent
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.callback_on_new_best = callback_on_new_best
        self.best_mean_reward = -np.inf
    
    def should_evaluate(self, episode):
        """Check if should evaluate."""
        return self.eval_freq > 0 and episode % self.eval_freq == 0
    
    def evaluate(self, episode):
        """Evaluate agent performance."""
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            self.agent.reset_state()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action = self.agent.select_action(obs, eval_mode=True)
                obs, reward, done = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        # Track best
        is_best = mean_reward > self.best_mean_reward
        if is_best:
            self.best_mean_reward = mean_reward
            if self.callback_on_new_best is not None:
                self.callback_on_new_best(suffix=f"_best_ep{episode}")
        
        print(f"  Eval ({self.n_eval_episodes} eps): "
              f"Mean Reward={mean_reward:.2f}, "
              f"Mean Length={mean_length:.1f}"
              f"{' [NEW BEST]' if is_best else ''}")
        
        return {
            'mean_reward': mean_reward,
            'mean_length': mean_length,
            'is_best': is_best
        }





class MetricsCallback:
    """Logs training metrics to file and Tensorboard."""
    
    def __init__(self, log_path='logs', name='metrics'):
        """
        Args:
            log_path: Directory to save metrics
            name: Name of metrics file
        """
        self.log_path = log_path
        self.metrics_file = os.path.join(log_path, f'{name}.json')
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_times': [],
            'world_model_losses': [],
            'actor_losses': [],
            'critic_losses': []
        }
        
        os.makedirs(log_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_path)
    
    def log_episode(self, episode, reward, length, duration, step=None, info=None):
        """Log episode metrics."""
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(length)
        self.metrics['episode_times'].append(duration)
        
        # Tensorboard
        if step is None:
            step = episode # Fallback if global step not provided
            
        self.writer.add_scalar('charts/episode_reward', reward, step)
        self.writer.add_scalar('charts/episode_length', length, step)
        self.writer.add_scalar('charts/episode_duration', duration, step)
        
        # W&B (if active)
        if wandb and wandb.run:
            wb_data = {
                'charts/episode_reward': reward,
                'charts/episode_length': length,
                'charts/episode_duration': duration,
                'charts/episode': episode
            }
            if info:
                if 'frags' in info: wb_data['gameplay/frags'] = info['frags']
                if 'health' in info: wb_data['gameplay/health_remaining'] = info['health']
                if 'ammo' in info: wb_data['gameplay/ammo_consumed'] = info['ammo']
            wandb.log(wb_data, step=step)

        # Detailed gameplay metrics if provided
        if info:
            if 'frags' in info:
                self.writer.add_scalar('gameplay/frags', info['frags'], step)
            if 'health' in info:
                self.writer.add_scalar('gameplay/health_remaining', info['health'], step)
            if 'ammo' in info:
                self.writer.add_scalar('gameplay/ammo_consumed', info['ammo'], step)
        
        # Log mean of recent 100
        if len(self.metrics['episode_rewards']) >= 100:
            mean_reward = np.mean(self.metrics['episode_rewards'][-100:])
            self.writer.add_scalar('charts/mean_episode_reward_100', mean_reward, step)

    def log_training(self, step, **kwargs):
        """Log training metrics."""
        # Log all kwargs to Tensorboard
        for key, value in kwargs.items():
            # Filter non-scalar values just in case
            if isinstance(value, (int, float, np.number)):
                # Clean up key name for TB (e.g., 'actor_loss' -> 'losses/actor_loss')
                if 'loss' in key:
                    tb_key = f"losses/{key}"
                elif key == 'fps':
                    tb_key = f"charts/{key}"
                else:
                    tb_key = f"train/{key}"
                self.writer.add_scalar(tb_key, value, step)
                
                # W&B (if active)
                if wandb and wandb.run:
                    wandb.log({tb_key: value}, step=step)
                
            # Store specific ones to metrics dict for JSON
            if key in ['image_loss', 'reward_loss', 'cont_loss', 'kl_loss']:
                 # Sum up WM loss? Or just store individual?
                 # JSON structure expects 'world_model_losses' list.
                 # Let's just track 'loss' (total loss) if available
                 pass
        
        # Backward compatibility for specific JSON lists
        if 'loss' in kwargs: # Total WM loss usually
            self.metrics['world_model_losses'].append(kwargs['loss'])
        if 'actor_loss' in kwargs:
            self.metrics['actor_losses'].append(kwargs['actor_loss'])
        if 'critic_loss' in kwargs:
            self.metrics['critic_losses'].append(kwargs['critic_loss'])
    
    def save(self):
        """Save metrics to file."""
        # Convert numpy types to python types for JSON
        def convert(o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o
            
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=convert)
            
    def close(self):
        """Close writer."""
        self.writer.close()
    
    def get_recent_stats(self, n=100):
        """Get statistics from recent episodes."""
        if len(self.metrics['episode_rewards']) == 0:
            return {}
        
        recent_rewards = self.metrics['episode_rewards'][-n:]
        recent_lengths = self.metrics['episode_lengths'][-n:]
        
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_length': np.mean(recent_lengths),
            'std_length': np.std(recent_lengths)
        }
