"""
Training callbacks for Dreamer V3.

Provides callbacks for video recording, checkpointing, evaluation, and metrics logging.
"""

import os
import json
import imageio
import numpy as np

from doom_agent.algorithms.dreamer_v3.doom_envs import DoomDreamerEnv


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
        # obs puede venir como (H, W, 1) float o uint8; normalizamos a uint8 HxW
        if obs is None:
            return np.zeros((64, 64), dtype=np.uint8)
        frame = obs
        if frame.ndim == 4:  # batch dim
            frame = frame[0]
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
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
                action = self.agent.select_action(obs, deterministic=self.deterministic)
                obs, reward, done = self.eval_env.step(action)

        if frames:
            save_file = os.path.join(self.save_path, f"{self.name_prefix}{suffix}.gif")
            # use duration instead of fps to avoid DeprecationWarning (duration is in ms)
            imageio.mimsave(save_file, frames, duration=1000/self.fps)
            print(f"  Saved video: {save_file}")


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



from torch.utils.tensorboard import SummaryWriter

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
    
    def log_episode(self, episode, reward, length, duration, step=None):
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
