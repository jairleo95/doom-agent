import torch
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import time
from functools import partial

from doom_agent.algorithms.dreamer.v3.replay_buffer import ReplayBuffer
from doom_agent.algorithms.dreamer.v3.doom_envs import DoomDreamerEnv
from doom_agent.algorithms.dreamer.v3.parallel_fix import Parallel, Damy

class RLDataset(IterableDataset):
    """
    Iterable Dataset that interacts with Environments to fill buffer and yields batches.
    """
    def __init__(self, buffer, train_envs, agent, actions, cfg, training_steps_per_epoch):
        self.buffer = buffer
        self.train_envs = train_envs
        self.agent = agent
        self.actions = actions
        self.cfg = cfg
        self.steps_per_epoch = training_steps_per_epoch
        
        # State management
        self.obs_list = [e.reset()() for e in self.train_envs]
        self.agent.reset_state()
        self.is_first_list = [True] * cfg.agent.n_envs
        self.env_step_counter = 0
        
        # Episode Tracking
        self.episode_rewards = [0.0] * cfg.agent.n_envs
        self.episode_lengths = [0] * cfg.agent.n_envs
        
        # Gameplay Stats
        self.episode_frags = [0.0] * cfg.agent.n_envs
        self.episode_health = [0.0] * cfg.agent.n_envs # Average health
        self.episode_ammo = [0.0] * cfg.agent.n_envs # Average ammo
        self.episode_steps = [0] * cfg.agent.n_envs # Same as lengths? Yes.
        
        self.pending_metrics = [] # List of completed episode stats

    def __iter__(self):
        """
        Yields batches for one epoch.
        Inside the loop, it also runs environment steps.
        """
        batch_count = 0
        while batch_count < self.steps_per_epoch:
            # 1. Environment Interaction Phase
            # Run roughly (train_every) env steps for every batch we yield
            # This maintains the ratio
            steps_needed = self.cfg.agent.train_every
            
            # Since we step n_envs at a time
            steps_taken = 0
            while steps_taken < steps_needed:
                obs_batch = np.stack(self.obs_list)
                # Select action using the agent (which is the LightningModule or inner agent)
                # Note: agent must be on correct device or CPU. Usually CPU for inference here.
                actions_vec = self.agent.select_action(obs_batch, is_first=self.is_first_list)
                if self.cfg.agent.n_envs == 1: actions_vec = [actions_vec]
                
                step_futures = [e.step(a) for e, a in zip(self.train_envs, actions_vec)]
                step_results = [f() for f in step_futures]
                
                for i, (next_obs, reward, done, info) in enumerate(step_results):
                    self.buffer.add(self.obs_list[i], actions_vec[i], reward, float(done), self.is_first_list[i])
                    
                    # Track metrics
                    self.episode_rewards[i] += reward
                    self.episode_lengths[i] += 1
                    
                    # Gameplay Stats (info has 'frags', 'health', 'ammo' at current step)
                    # We might want max frags (cumulative) and avg health/ammo.
                    # Frags are cumulative in the env state usually, but DoomDreamerEnv returns current frag count.
                    # Actually DoomDreamerEnv `info['frags']` is `last_frag_count` (total in episode so far).
                    # So we take the MAX (or last value) for frags.
                    # For health/ammo we usually want average over episode.
                    
                    current_frags = info.get('frags', 0)
                    self.episode_frags[i] = current_frags # Keep updating since it's cumulative (total kills)
                    
                    self.episode_health[i] += info.get('health', 0)
                    self.episode_ammo[i] += info.get('ammo', 0)
                    
                    if done:
                        # Calculate averages
                        avg_health = self.episode_health[i] / max(1, self.episode_lengths[i])
                        avg_ammo = self.episode_ammo[i] / max(1, self.episode_lengths[i])
                        
                        self.pending_metrics.append({
                            'return': self.episode_rewards[i],
                            'length': self.episode_lengths[i],
                            'frags': self.episode_frags[i],
                            'health_avg': avg_health,
                            'ammo_avg': avg_ammo
                        })
                        
                        # Reset
                        self.episode_rewards[i] = 0.0
                        self.episode_lengths[i] = 0
                        self.episode_frags[i] = 0.0
                        self.episode_health[i] = 0.0
                        self.episode_ammo[i] = 0.0
                        
                    self.obs_list[i], self.is_first_list[i] = (next_obs, done) if not done else (self.train_envs[i].reset()(), True)
                    
                steps_taken += self.cfg.agent.n_envs
                self.env_step_counter += self.cfg.agent.n_envs

            # 2. Training Batch Generation Phase
            if len(self.buffer) >= self.cfg.agent.batch_size * self.cfg.agent.batch_length:
                # Can flip logic
                can_flip = len(self.actions) == 12
                do_flip = can_flip and (np.random.random() < 0.5)
                
                batch = self.buffer.sample(self.cfg.agent.batch_size, horizontal_flip=do_flip)
                if batch:
                    # Flip actions if needed (buffer handles obs flip)
                    if do_flip:
                        from doom_agent.algorithms.dreamer.v3.utils import flip_actions
                        batch['action'] = flip_actions(batch['action'])
                    
                    # Attach metrics if any episodes finished
                    if self.pending_metrics:
                        batch['epoch_metrics'] = self.pending_metrics
                        self.pending_metrics = [] # Clear
                        # We attach as list of dicts. 
                        # PL transfer_batch_to_device might complain if it tries to move list of dicts.
                        # We handle this in transfer_batch_to_device
                        
                    yield batch
                    batch_count += 1
            else:
                # Buffer not full enough, just continue collecting without yielding
                # In strict Iterator protocol we should yield something or wait.
                # Since this is "train logic", we loop until we have enough data.
                continue

class DoomDataModule(pl.LightningDataModule):
    def __init__(self, cfg, agent, actions, stage_config):
        super().__init__()
        self.cfg = cfg
        self.agent = agent
        self.actions = actions
        self.stage_config = stage_config
        
        # Factory method for envs
        # Resolve scenario: stage override -> global scenario -> fallback
        scenario_cfg = stage_config.scenario if stage_config.scenario else cfg.scenario.scenario_name + ".cfg"
        
        self.env_factory = partial(
             DoomDreamerEnv,
             scenario=scenario_cfg,
             actions=actions,
             frame_skip=stage_config.frame_skip,
             # ... other params ...
             doom_skill=stage_config.doom_skill,
             living_reward=stage_config.living_reward,
             health_penalty=stage_config.health_penalty,
             ammo_penalty=stage_config.ammo_penalty,
             frag_bonus=stage_config.frag_bonus,
             obs_shape=(64, 64, 3)
        )
        
        # Replay Buffer
        self.buffer = ReplayBuffer(
            capacity=cfg.agent.get('buffer_capacity', 1_000_000),
            sequence_length=cfg.agent.batch_length,
            obs_shape=(64, 64, 3)
        )
        
        self.train_envs = []

    def setup(self, stage=None):
        # Create Environments
        if not self.train_envs:
            if self.cfg.agent.n_envs > 1:
                # We need to pickle factories, so we need clean partials or classes. 
                # Reusing Parallel logic from trainer.py
                # This needs to run in main process usually.
                self.train_envs = [Parallel(partial(self.env_factory, window_visible=False), "process") for _ in range(self.cfg.agent.n_envs)]
            else:
                # Single env
                self.train_envs = [Damy(self.env_factory(window_visible=False))]
                
        # Prefill if needed (Main process)
        if len(self.buffer) == 0:
             self._prefill()

    def _prefill(self):
        print(f"Prefilling buffer with {self.cfg.agent.prefill_steps} steps...")
        # (Similar logic to trainer.py _prefill, using self.train_envs)
        # Simplified for brevity in this step, assume buffer gets filled
        pass # TODO: Add prefill logic here or in RLDataset init first run

    def train_dataloader(self):
        # Determine how many "batches" represent an epoch
        # In RL, "epoch" is arbitrary. Let's say 1 epoch = 1000 updates.
        steps_per_epoch = 1000 
        
        dataset = RLDataset(
            self.buffer, 
            self.train_envs, 
            self.agent, 
            self.actions, 
            self.cfg,
            training_steps_per_epoch=steps_per_epoch
        )
        
        # Num workers must be 0 because we handle parallelism inside the envs (Parallel wrapper)
        # and we modify the buffer in-place.
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def teardown(self, stage=None):
        for e in self.train_envs:
            e.close()
