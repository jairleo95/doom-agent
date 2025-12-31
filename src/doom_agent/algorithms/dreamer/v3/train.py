import os
import sys
import torch
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# Allow importing local modules
sys.path.append(str(Path(__file__).resolve().parent))

from doom_agent.algorithms.dreamer.v3.experiment import ExperimentManager
from doom_agent.algorithms.dreamer.v3.curriculum import Curriculum, Stage
from doom_agent.algorithms.dreamer.v3.doom_envs import universal_actions, deadly_corridor_actions, defend_actions

from doom_agent.algorithms.dreamer.v3.lightning_module import DoomLightningModule
from doom_agent.algorithms.dreamer.v3.lightning_datamodule import DoomDataModule
from doom_agent.algorithms.dreamer.v3.callbacks import VideoRecorderCallback, ImaginationVideoCallback, EvalCallback

# Disable audio at the OS level
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['ALSOFT_DRIVERS'] = 'null'

def get_action_set(scenario_name):
    """Get action set based on scenario."""
    if 'deathmatch' in scenario_name or 'universal' in scenario_name:
        return universal_actions()
    elif 'deadly_corridor' in scenario_name:
        return deadly_corridor_actions()
    elif 'defend_the_center' in scenario_name:
        return defend_actions()
    return universal_actions()

def create_curriculum(cfg: DictConfig):
    """Convert Hydra config to Curriculum object."""
    stages = [Stage(**s) for s in cfg.scenario.curriculum.stages]
    return Curriculum(
        name=cfg.scenario.name,
        scenario=cfg.scenario.scenario_name + ".cfg",
        stages=stages
    )

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Setup Experiment
    exp = ExperimentManager(cfg)
    curriculum = create_curriculum(cfg)
    actions = get_action_set(cfg.scenario.name)
    
    # Track metadata
    exp.save_config(curriculum)
    exp.update_manifest(curriculum.name)
    # Note: We rely on PL WandbLogger instead of manual init, but exp.init_wandb() does setup.
    # We can skip exp.init_wandb() if we use WandbLogger exclusively, but generic setup is good.
    # Let's use PL Loggers.
    
    # Configure Performance
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('medium') # For TensorCores on Ampere+

    # 1. Initialize Lightning Module (Model)
    model = DoomLightningModule(cfg, actions, run_dir=exp.log_dir)
    
    # Reload if requested
    if cfg.resume:
        print(f"Resuming from {cfg.resume}")
        # We can load weights here or let trainer resume via 'ckpt_path' arg in fit
        # model = DoomLightningModule.load_from_checkpoint(cfg.resume, cfg=cfg, actions=actions, run_dir=exp.log_dir)
        pass 

    # 2. Iterate Curriculum Stages
    global_step_offset = 0
    
    for idx, stage in enumerate(curriculum.stages):
        if idx < cfg.start_stage:
            print(f"Skipping Stage {idx}: {stage.name}")
            global_step_offset += stage.timesteps
            continue
            
        print(f"\n=== Running Stage {idx}: {stage.name} ===")
        print(f"Config: Skill={stage.doom_skill}, Reward={stage.living_reward}, Timesteps={stage.timesteps}")
        
        # Data Module for this stage
        dm = DoomDataModule(cfg, model.agent, actions, stage_config=stage)
        
        # Callbacks
        stage_ckpt_dir = exp.get_stage_ckpt_dir(stage.name)
        stage_video_dir = exp.get_video_dir(stage.name)
        
        callbacks = [
            ModelCheckpoint(
                dirpath=stage_ckpt_dir,
                filename="dreamer_{step}",
                every_n_train_steps=50_000,
                save_top_k=-1 # Save all requested
            ),
            VideoRecorderCallback(
                eval_env=dm.env_factory(window_visible=False), # Create separate eval env
                save_path=str(stage_video_dir),
                name_prefix=f"dreamer_{stage.name}",
                render_freq=cfg.video_freq,
                deterministic=True
            ),
            EvalCallback(
                eval_env=dm.env_factory(window_visible=False),
                eval_freq=cfg.video_freq, # Sync with video
                n_eval_episodes=5
            ),
            ImaginationVideoCallback(render_freq=cfg.video_freq or 1000)
        ]
        
        # Loggers
        loggers = [TensorBoardLogger(save_dir=exp.get_stage_log_dir(stage.name), name="tb")]
        if cfg.wandb.enabled:
            loggers.append(WandbLogger(
                project=cfg.wandb.project, 
                entity=cfg.wandb.entity,
                name=f"{exp.run_id}_{stage.name}", 
                save_dir=exp.log_dir,
                group=cfg.wandb.group
            ))

        # Trainer
        # Calculate max_steps logic. PL is cumulative.
        # But we create a NEW trainer for each stage to reset step counts simpler, 
        # OR we manage accumulation.
        # Simpler to create new trainer, BUT we want global_step to increase.
        # PL doesn't support easy "continue global step" if new trainer.
        # We will use ONE trainer? No, callbacks/loggers are stage-specific.
        # We will use ONE trainer for simplicity of resources, but updating callbacks is hard.
        # We will use NEW trainer per stage, but realize global_step starts at 0 for each stage in logs (Lap Step).
        # This is actually fine for curriculum.
        
        trainer = pl.Trainer(
            accelerator="gpu" if cfg.device == "cuda" else "cpu",
            devices=1,
            # PL counts optimization steps (batches). 
            # Total environment steps / env_steps_per_batch = total_batches
            # In Dreamer, we train once every 'train_every' environment steps.
            max_steps=stage.timesteps // cfg.agent.train_every, 
            callbacks=callbacks,
            logger=loggers,
            default_root_dir=exp.log_dir,
            enable_progress_bar=True,
            limit_val_batches=0, # We handle eval via callback
            precision="16-mixed" if cfg.device == "cuda" else 32, # Enable AMP automatically!
        )
        
        # Run
        trainer.fit(model, datamodule=dm, ckpt_path=cfg.resume if idx == cfg.start_stage and idx > 0 else None)
        
        # Save Final Stage Model
        final_path = stage_ckpt_dir / f"dreamer_{stage.name}_final.pt"
        trainer.save_checkpoint(final_path)
        
        # Cleanup
        dm.teardown()
        if cfg.wandb.enabled:
             wandb.finish() # Close run for this stage to avoid accumulation issues or just reuse
             
    print("Training Complete.")

if __name__ == "__main__":
    main()
