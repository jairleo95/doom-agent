import numpy as np
import pytorch_lightning as pl

class EvalCallback(pl.Callback):
    """Evaluates agent performance periodically."""
    
    def __init__(
        self,
        eval_env,
        eval_freq=50,  # Evaluate every N steps (not episodes in PL context usually, but we use logic)
        n_eval_episodes=5,
        callback_on_new_best=None
    ):
        """
        Args:
            eval_env: Environment for evaluation
            eval_freq: Evaluate every N global steps
            n_eval_episodes: Number of episodes to evaluate
            callback_on_new_best: Callback to trigger on new best model
        """
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.callback_on_new_best = callback_on_new_best
        self.best_mean_reward = -np.inf
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # We use strict step frequency to align with PL
        if self.eval_freq > 0 and trainer.global_step > 0 and trainer.global_step % self.eval_freq == 0:
            self.evaluate(trainer, pl_module)

    def evaluate(self, trainer, pl_module):
        """Evaluate agent performance."""
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            pl_module.reset_state()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action = pl_module.select_action(obs, eval_mode=True)
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
                # We assume callback_on_new_best takes (pl_module, suffix)
                self.callback_on_new_best(pl_module, suffix=f"_best_step_{trainer.global_step}")
        
        print(f"  Eval ({self.n_eval_episodes} eps): "
              f"Mean Reward={mean_reward:.2f}, "
              f"Mean Length={mean_length:.1f}"
              f"{' [NEW BEST]' if is_best else ''}")
        
        # Log to PL logger
        pl_module.log("eval/mean_reward", mean_reward, prog_bar=True)
        pl_module.log("eval/mean_length", mean_length, prog_bar=True)
        
        return {
            'mean_reward': mean_reward,
            'mean_length': mean_length,
            'is_best': is_best
        }
