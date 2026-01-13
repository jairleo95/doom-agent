import numpy as np

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
                obs, reward, done, info = self.eval_env.step(action)
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
