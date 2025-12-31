import os
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None

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
