"""
Training callbacks for Dreamer V3.

Provides callbacks for video recording, checkpointing, evaluation, and metrics logging.
"""

import os
from doom_agent.algorithms.dreamer_v3.doom_envs import DoomDreamerEnv
import numpy as np


class VideoRecorderCallback:
    """Records episode videos as GIFs."""
    
    def __init__(
        self,
        eval_env,
        agent,
        save_path='videos',
        name_prefix='dreamer_v3',
        record_freq=100,  # Record every N episodes
        n_episodes=1
    ):
        """
        Args:
            eval_env: Environment to record
            agent: DreamerV3Agent instance
            save_path: Directory to save videos
            name_prefix: Prefix for video filenames
            record_freq: Record every N episodes
            n_episodes: Number of episodes to record
        """
        self.eval_env = eval_env
        self.agent = agent
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.record_freq = record_freq
        self.n_episodes = n_episodes
        
        os.makedirs(save_path, exist_ok=True)
    
    def should_record(self, episode):
        """Check if should record this episode."""
        return self.record_freq > 0 and episode % self.record_freq == 0
    
    def record_video(self, episode):
        """Record video of agent playing."""
        frames = []
        
        for _ in range(self.n_episodes):
            obs = self.eval_env.reset()
            self.agent.reset_state()
            done = False
            
            while not done:
                # Convert observation to frame for video
                frame = (obs[0] * 255).astype(np.uint8)
                frames.append(frame)
                
                # Get action from agent
                action = self.agent.select_action(obs)
                obs, reward, done = self.eval_env.step(action)
        
        # Save as GIF
        if len(frames) > 0:
            save_file = os.path.join(
                self.save_path, 
                f"{self.name_prefix}_ep{episode}.gif"
            )
            imageio.mimsave(save_file, frames, fps=30)
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
        n_eval_episodes=5
    ):
        """
        Args:
            eval_env: Environment for evaluation
            agent: DreamerV3Agent instance
            eval_freq: Evaluate every N episodes
            n_eval_episodes: Number of episodes to evaluate
        """
        self.eval_env = eval_env
        self.agent = agent
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
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
                action = self.agent.select_action(obs)
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
    """Logs training metrics to file."""
    
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
    
    def log_episode(self, episode, reward, length, duration):
        """Log episode metrics."""
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(length)
        self.metrics['episode_times'].append(duration)
    
    def log_training(self, world_model_loss=None, actor_loss=None, critic_loss=None):
        """Log training metrics."""
        if world_model_loss is not None:
            self.metrics['world_model_losses'].append(world_model_loss)
        if actor_loss is not None:
            self.metrics['actor_losses'].append(actor_loss)
        if critic_loss is not None:
            self.metrics['critic_losses'].append(critic_loss)
    
    def save(self):
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
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
