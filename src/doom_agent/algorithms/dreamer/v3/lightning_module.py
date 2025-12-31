import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import OmegaConf
import numpy as np
from doom_agent.algorithms.dreamer.v3.agent import DreamerV3Agent

class DoomLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for DreamerV3Agent.
    """

    def __init__(self, cfg, actions, run_dir):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.actions = actions
        self.run_dir = run_dir
        
        # Initialize Agent
        # We need to construct the config dictionary expected by DreamerV3Agent
        agent_config = OmegaConf.to_container(cfg.agent, resolve=True)
        agent_config['num_actions'] = len(actions)
        agent_config['action_dim'] = len(actions)
        agent_config['device'] = cfg.device
        agent_config['compile'] = cfg.get('compile', False)
        
        self.agent = DreamerV3Agent(agent_config, run_dir=run_dir)
        # Register the inner Dreamer model as a submodule so PL tracks parameters/device
        self.dreamer = self.agent.agent
        
        # Important: PL controls the device. We need to ensure the agent 
        # uses the device PL assigns to this module.
        # DreamerV3Agent internal .to(device) is called in __init__, 
        # but we might need to sync it later.
        
        # We manually handle optimization inside agent.train_step()
        self.automatic_optimization = False
        self.manual_global_step = 0

    def training_step(self, batch, batch_idx):
        """
        Execute one training step.
        Batch is provided by the RLDataset (replay buffer samples).
        """
        # The batch is already a dictionary of tensors from the ReplayBuffer
        metrics = self.agent.train_step(batch)
        
        # Log metrics
        # Dreamer returns raw values (often numpy), we log them
        # Ensure conversion to supported types
        for k, v in metrics.items():
            if hasattr(v, 'mean'):
                 # It's likely a tensor or numpy array
                 # If it's 0-d, mean works too.
                 v = v.mean()
            
            if hasattr(v, 'item'):
                v = v.item()
                
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=False)

        # Log immediate batch reward statistics (for dense reward feedback)
        if 'reward' in batch:
            reward_mean = batch['reward'].float().mean().item()
            self.log("train/reward_mean", reward_mean, on_step=True, on_epoch=False, prog_bar=True)
            
        # We return None because optimization happens inside agent.train_step
        # We are using Manual Optimization conceptually, but since we don't expose optimizers, 
        # we treat the agent as a black box operation.
        
        
        # Log episode metrics if present
        if 'epoch_metrics' in batch:
            metrics_list = batch['epoch_metrics']
            # metrics_list is a list of dicts: [{'return': r, 'length': l}, ...]
            avg_return = np.mean([m['return'] for m in metrics_list])
            avg_length = np.mean([m['length'] for m in metrics_list])
            
            self.log("episode/return", avg_return, on_step=True, on_epoch=False, prog_bar=True)
            self.log("episode/length", avg_length, on_step=True, on_epoch=False, prog_bar=True)
            self.log("episode/count", float(len(metrics_list)), on_step=True, on_epoch=False)
            
            # Gameplay stats
            if 'frags' in metrics_list[0]:
                 avg_frags = np.mean([m.get('frags', 0) for m in metrics_list])
                 self.log("episode/frags", avg_frags, on_step=True, on_epoch=False, prog_bar=True)
                 
            if 'health_avg' in metrics_list[0]:
                 avg_health = np.mean([m.get('health_avg', 0) for m in metrics_list])
                 self.log("gameplay/health_avg", avg_health, on_step=True, on_epoch=False)

            if 'ammo_avg' in metrics_list[0]:
                 avg_ammo = np.mean([m.get('ammo_avg', 0) for m in metrics_list])
                 self.log("gameplay/ammo_avg", avg_ammo, on_step=True, on_epoch=False)

        # FPS calculation (approximate based on batch processing speed)
        # PL logs `batch/s` automatically. We can log `env_steps/s`
        # env_steps_per_batch = batch_length * batch_size?
        # No, RLDataset collects `train_every` env steps per update roughly?
        # Actually, if we use `train_every` logic (ratio), we consume ~batch_size*batch_len/ratio steps?
        # But we actually STEP the environment `train_every` times per batch yield in DataModule logic.
        # So SPS = Batches/Sec * train_every. Not quite.
        # Let's rely on PL's `throughput`.
        
        self.manual_global_step += 1
        return None

    @property
    def global_step_custom(self):
        return self.manual_global_step

    def configure_optimizers(self):
        # We don't expose optimizers to PL yet because they are buried in the internal library.
        # The internal loop handles zero_grad/step.
        return []

    def load_state_dict(self, state_dict, strict=True):
        # Intercept loading to use agent's loader if needed, 
        # currently standard PL loading works if parameters are registered.
        # However, DreamerV3Agent holds 'self.agent' which is a nn.Module.
        # PL will call this on self.
        super().load_state_dict(state_dict, strict=strict)

    def on_save_checkpoint(self, checkpoint):
        # Helper to ensure internal state is saved if it's not a direct submodule
        # self.agent.agent IS a nn.Module, so it should be in state_dict automatically.
        pass

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # ReplayBuffer samples are tensors. PL creates a batch.
        # We just move the dict of tensors to device.
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            # Skip non-tensors (like 'epoch_metrics' list)
        return batch

    # Proxy for action selection
    def select_action(self, obs, eval_mode=False, is_first=None):
        return self.agent.select_action(obs, eval_mode=eval_mode, is_first=is_first)
    
    def on_train_start(self):
        """
        Sync the internal agent's device with the LightningModule's device.
        PL moves the parameters, but we need to update the 'self.device' attribute
        in the adapter so it moves input tensors correctly.
        """
        self.agent.device = self.device
        # Ensure the internal Dreamer logic also knows the device if it caches it
        # (DreamerV3Agent takes care of this usually by using self.device)
        print(f"DreamerV3Agent synced to device: {self.device}")

    def reset_state(self):
        self.agent.reset_state()
