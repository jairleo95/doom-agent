import torch
from pathlib import Path
import pytorch_lightning as pl

try:
    import wandb
except ImportError:
    wandb = None

class ImaginationVideoCallback(pl.Callback):
    """Callback to record 'imagination' videos (world model predictions) to Logger."""
    
    def __init__(self, render_freq=1000):
        self.render_freq = render_freq
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = getattr(pl_module, 'global_step_custom', trainer.global_step)
        if self.render_freq > 0 and global_step % self.render_freq == 0:
            self.record_imagination(trainer, pl_module, batch)
            
    def record_imagination(self, trainer, pl_module, batch):
        """Record model predictions for a given batch of data."""
        global_step = getattr(pl_module, 'global_step_custom', trainer.global_step)
        # agent.agent is the actual Dreamer model
        with torch.no_grad():
            try:
                # video_pred returns (Truth, Prediction, Error) concatenated
                # Shape is (Batch, Time, H, W*3, C)
                # Correct path: agent.agent._wm.video_pred
                video = pl_module.agent.agent._wm.video_pred(batch)
                
                # Convert to (T, C, H, W) for TensorBoard (takes one video at a time)
                # We take the first 1 video in the batch
                video_tb = video[0] # (T, H, W, C)
                video_tb = video_tb.permute(0, 3, 1, 2) # (T, C, H, W)
                
                # TensorBoard wants (N, T, C, H, W) where N is number of videos
                video_tb = video_tb.unsqueeze(0)
                
                # trainer.logger.experiment is usually the SummaryWriter for TensorBoardLogger
                if hasattr(trainer.logger, 'experiment'):
                    if hasattr(trainer.logger.experiment, 'add_video'):
                         trainer.logger.experiment.add_video("imagination/truth_pred_error", video_tb, trainer.global_step, fps=15)
                
                # W&B Visualization (if active and global wandb is available)
                if wandb and wandb.run:
                    # wandb.Video takes (T, H, W, C) or (T, C, H, W)
                    video_wb = video[0].cpu().numpy()
                    wandb.log({
                        "imagination/truth_pred_error": wandb.Video(video_wb, fps=15, format="gif")
                    }, step=trainer.global_step)

            except Exception as e:
                print(f"Warning: Failed to log imagination video: {e}")
