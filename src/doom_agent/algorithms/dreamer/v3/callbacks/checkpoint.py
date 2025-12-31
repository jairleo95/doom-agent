import os

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
