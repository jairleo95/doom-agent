import time
import json
import numpy as np
import torch
import wandb
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from functools import partial

from doom_agent.algorithms.dreamer.v3.agent import DreamerV3Agent
from doom_agent.algorithms.dreamer.v3.doom_envs import DoomDreamerEnv
from doom_agent.algorithms.dreamer.v3.replay_buffer import ReplayBuffer
from doom_agent.algorithms.dreamer.v3.callbacks import (
    VideoRecorderCallback,
    EvalCallback,
    MetricsCallback,
    ImaginationVideoCallback
)
from doom_agent.algorithms.dreamer.v3.parallel_fix import Parallel, Damy
from doom_agent.algorithms.dreamer.v3.utils import flip_actions, format_time

class DreamerV3Trainer:
    """Orchestrates DreamerV3 training on VizDoom environments."""
    
    def __init__(self, cfg: DictConfig, experiment, curriculum, actions):
        self.cfg = cfg
        self.exp = experiment
        self.curriculum = curriculum
        self.actions = actions
        self.device = cfg.device
        
        self.global_step = 0
        self.best_eval_reward = -float('inf')
        
        # Agent initialization
        agent_config = OmegaConf.to_container(cfg.agent, resolve=True)
        agent_config['num_actions'] = len(actions)
        agent_config['action_dim'] = len(actions)
        agent_config['device'] = self.device
        agent_config['compile'] = cfg.get('compile', False)
        self.agent = DreamerV3Agent(agent_config, run_dir=self.exp.log_dir)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            capacity=cfg.agent.get('buffer_capacity', 1_000_000),
            sequence_length=agent_config['batch_length'],
            obs_shape=tuple(agent_config['obs_shape'])
        )
        
        self.total_curriculum_steps = sum(s.timesteps for s in curriculum.stages)

    def _make_env_factory(self, scenario_cfg, stage_config, idx=0):
        """Pickle-safe env factory."""
        return DoomDreamerEnv(
            scenario=scenario_cfg,
            actions=self.actions,
            frame_skip=stage_config.frame_skip,
            window_visible=self.cfg.visualize if (self.cfg.visualize and idx == 0) else False,
            doom_skill=stage_config.doom_skill,
            living_reward=stage_config.living_reward,
            health_penalty=stage_config.health_penalty,
            ammo_penalty=stage_config.ammo_penalty,
            frag_bonus=stage_config.frag_bonus,
            obs_shape=(64, 64, 3)
        )

    def run(self):
        """Run the full curriculum training."""
        if self.cfg.resume:
            print(f"Loading checkpoint from: {self.cfg.resume}")
            self.agent.load(self.cfg.resume)

        for idx, stage in enumerate(self.curriculum.stages):
            if idx < self.cfg.start_stage:
                print(f"Skipping Stage {idx}: {stage.name}")
                continue
                
            self._run_stage(idx, stage)
            
        print("Training Complete.")

    def _run_stage(self, idx, stage):
        """Execute a single curriculum stage."""
        print(f"\n=== Running Stage {idx}: {stage.name} ===")
        print(f"Config: Skill={stage.doom_skill}, Reward={stage.living_reward}, Timesteps={stage.timesteps}")
        
        stage_start_time = time.time()
        scenario_cfg = stage.scenario or self.curriculum.scenario
        
        # Envs setup
        if self.cfg.agent.n_envs > 1:
            train_envs = [Parallel(partial(self._make_env_factory, scenario_cfg, stage, i), "process") for i in range(self.cfg.agent.n_envs)]
        else:
            train_envs = [Damy(self._make_env_factory(scenario_cfg, stage))]
            
        eval_env = self._make_env_factory(scenario_cfg, stage)
        
        # Callbacks
        stage_ckpt_dir = self.exp.get_stage_ckpt_dir(stage.name)
        log_dir = self.exp.get_stage_log_dir(stage.name)
        
        metrics_cb = MetricsCallback(log_path=str(log_dir), name='metrics')
        video_cb = VideoRecorderCallback(
            eval_env=eval_env, agent=self.agent,
            save_path=str(self.exp.get_video_dir(stage.name)),
            name_prefix=f"dreamer_{stage.name}",
            render_freq=self.cfg.video_freq, n_eval_episodes=1, deterministic=True
        )
        eval_cb = EvalCallback(
            eval_env=eval_env, agent=self.agent, eval_freq=10,
            n_eval_episodes=3, callback_on_new_best=video_cb.record_video
        )
        imag_video_cb = ImaginationVideoCallback(
            agent=self.agent, log_dir=log_dir, render_freq=self.cfg.video_freq or 1000
        )

        # Stage Training Loop
        self._train_loop(stage, train_envs, eval_cb, metrics_cb, imag_video_cb, video_cb, stage_ckpt_dir)
        
        # Cleanup & Finalize Stage
        duration = time.time() - stage_start_time
        final_path = stage_ckpt_dir / f"dreamer_{stage.name}_final.pt"
        self.agent.save(str(final_path))
        self._handle_stage_final_artifact(stage.name, final_path)
        
        self._save_stage_results(stage.name, duration, final_path)
            
        for e in train_envs: e.close()
        eval_env.close()
        metrics_cb.save()

    def _train_loop(self, stage, train_envs, eval_cb, metrics_cb, imag_video_cb, video_cb, stage_ckpt_dir):
        """Main training loop for a stage."""
        if self.global_step == 0 and not self.cfg.resume:
            self._prefill(train_envs)

        stage_step = 0
        obs_list = [e.reset()() for e in train_envs]
        self.agent.reset_state()
        is_first_list = [True] * self.cfg.agent.n_envs
        episode_count = 0
        
        env_ep_rewards = [0.0] * self.cfg.agent.n_envs
        env_ep_lengths = [0] * self.cfg.agent.n_envs
        env_ep_start_times = [time.time()] * self.cfg.agent.n_envs
        
        last_log = (time.time(), self.global_step)
        last_eval = (time.time(), self.global_step)
        last_imag_log_step = self.global_step
        train_counter = 0
        ema_fps = None

        while stage_step < stage.timesteps:
            obs_batch = np.stack(obs_list)
            actions_vec = self.agent.select_action(obs_batch, is_first=is_first_list)
            if self.cfg.agent.n_envs == 1: actions_vec = [actions_vec]
            
            step_futures = [e.step(a) for e, a in zip(train_envs, actions_vec)]
            step_results = [f() for f in step_futures]
            
            for i, (next_obs, reward, done) in enumerate(step_results):
                self.replay_buffer.add(obs_list[i], actions_vec[i], reward, float(done), is_first_list[i])
                env_ep_rewards[i] += reward
                env_ep_lengths[i] += 1
                obs_list[i], is_first_list[i] = next_obs, done
                
                if done:
                    episode_count += 1
                    ep_duration = time.time() - env_ep_start_times[i]
                    info = self._get_env_info(train_envs[i])
                    metrics_cb.log_episode(episode_count, env_ep_rewards[i], env_ep_lengths[i], ep_duration, step=self.global_step, info=info)
                    
                    if eval_cb.should_evaluate(episode_count):
                        eval_res = eval_cb.evaluate(self.global_step)
                        last_eval, ema_fps = self._log_eval_stats(eval_res, metrics_cb, last_eval, ema_fps)
                        self._handle_best_model_artifact(eval_res['mean_reward'], stage.name)

                    obs_list[i] = train_envs[i].reset()()
                    is_first_list[i], env_ep_rewards[i], env_ep_lengths[i], env_ep_start_times[i] = True, 0.0, 0, time.time()
            
            stage_step += self.cfg.agent.n_envs
            self.global_step += self.cfg.agent.n_envs
            
            # Periodic Logging
            if self.global_step >= last_log[1] + 100:
                last_log = self._log_periodic(stage, stage_step, last_log, metrics_cb)
            
            # Training
            train_counter += self.cfg.agent.n_envs
            if train_counter >= self.cfg.agent.train_every:
                self._train_step(train_counter, imag_video_cb, metrics_cb)
                train_counter %= self.cfg.agent.train_every
            
            if video_cb.should_record(self.global_step): 
                video_cb.record_video(suffix=f"_step_{self.global_step}")
            if self.global_step % 50_000 == 0:
                self.agent.save(str(stage_ckpt_dir / f"dreamer_{stage.name}_{self.global_step}.pt"))

    def _prefill(self, train_envs):
        print(f"Prefilling buffer with {self.cfg.agent.prefill_steps} steps...")
        obs_list = [e.reset()() for e in train_envs]
        self.agent.reset_state()
        is_first_list = [True] * self.cfg.agent.n_envs
        steps_done = 0
        while steps_done < self.cfg.agent.prefill_steps:
            actions_vec = [np.random.randint(0, len(self.actions)) for _ in range(self.cfg.agent.n_envs)]
            results = [e.step(a)() for e, a in zip(train_envs, actions_vec)]
            for i, (next_obs, reward, done) in enumerate(results):
                self.replay_buffer.add(obs_list[i], actions_vec[i], reward, float(done), is_first_list[i])
                obs_list[i], is_first_list[i] = (next_obs, done) if not done else (train_envs[i].reset()(), True)
            steps_done += self.cfg.agent.n_envs
        self.agent.reset_state()

    def _train_step(self, counter, imag_video_cb, metrics_cb):
        if len(self.replay_buffer) < self.cfg.agent.batch_size * self.cfg.agent.batch_length: return
        num_batches = (counter // self.cfg.agent.train_every) * self.cfg.agent.train_steps
        # Only enable horizontal flip for universal/deathmatch sets (length 12)
        # In a real scenario, we should check scenario name or a specific config flag.
        can_flip = len(self.actions) == 12
        
        for _ in range(num_batches):
            do_flip = can_flip and (np.random.random() < 0.5)
            batch = self.replay_buffer.sample(self.cfg.agent.batch_size, horizontal_flip=do_flip)
            if batch:
                if do_flip: batch['action'] = flip_actions(batch['action'])
                metrics = self.agent.train_step(batch)
                metrics_cb.log_training(self.global_step, **metrics)
                if imag_video_cb and self.global_step % imag_video_cb.render_freq == 0:
                    imag_video_cb.record_imagination(self.global_step, batch)

    def _get_env_info(self, env):
        info = { 'frags': getattr(env, 'last_frag_count', 0), 'health': getattr(env, 'last_health', 0), 'ammo': getattr(env, 'last_ammo', 0) }
        return {k: (v() if hasattr(v, '__call__') else v) for k, v in info.items()}

    def _log_periodic(self, stage, stage_step, last_log, metrics_cb):
        t_diff = time.time() - last_log[0]
        s_diff = self.global_step - last_log[1]
        fps = s_diff / t_diff if t_diff > 0 else 0
        stage_pct = (stage_step / stage.timesteps) * 100
        global_pct = (self.global_step / self.total_curriculum_steps) * 100
        print(f"[{stage.name}] Step {stage_step}/{stage.timesteps} ({stage_pct:.1f}%) - Global {self.global_step} - FPS: {fps:.2f}")
        metrics_cb.log_training(self.global_step, fps=fps)
        return time.time(), self.global_step

    def _log_eval_stats(self, eval_res, metrics_cb, last_eval, ema_fps):
        metrics_cb.log_training(self.global_step, eval_mean_reward=eval_res['mean_reward'], eval_mean_length=eval_res['mean_length'])
        curr_t = time.time()
        eval_t, eval_s = curr_t - last_eval[0], self.global_step - last_eval[1]
        if eval_t > 0:
            fps = eval_s / eval_t
            ema_fps = fps if ema_fps is None else 0.3 * fps + 0.7 * ema_fps
            rem_s = self.total_curriculum_steps - self.global_step
            eta = rem_s / ema_fps if ema_fps > 0 else 0
            print(f"  Lap Stats: FPS={fps:.2f}, EMA_FPS={ema_fps:.2f}, ETA={format_time(eta)}")
        return (curr_t, self.global_step), ema_fps

    def _handle_best_model_artifact(self, reward, stage_name):
        if self.cfg.wandb.enabled and self.cfg.wandb.save_artifacts:
            if reward > self.best_eval_reward:
                self.best_eval_reward = reward
                best_path = self.exp.ckpt_dir / "best_model.pt"
                self.agent.save(str(best_path))
                self._upload_artifact(best_path, f"{self.exp.run_id}_best_model", "model", f"Best reward: {reward:.2f}", {"reward": reward, "step": self.global_step, "stage": stage_name})

    def _handle_stage_final_artifact(self, stage_name, path):
        if self.cfg.wandb.enabled and self.cfg.wandb.save_artifacts:
            self._upload_artifact(path, f"{self.exp.run_id}_{stage_name}_final", "model", f"Final {stage_name}", {"stage": stage_name, "step": self.global_step})

    def _upload_artifact(self, path, name, type, desc, meta):
        art = wandb.Artifact(name=name, type=type, description=desc, metadata=meta)
        art.add_file(str(path))
        wandb.log_artifact(art)

    def _save_stage_results(self, stage_name, duration, final_path):
        res = {"stage": stage_name, "duration_s": duration, "final_model": str(final_path), "global_step": self.global_step}
        with open(self.exp.log_dir / f"result_{stage_name}.json", "w") as f:
            json.dump(res, f, indent=4)
