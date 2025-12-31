from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None

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
