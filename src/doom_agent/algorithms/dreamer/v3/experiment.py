import os
import json
import csv
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from omegaconf import OmegaConf, DictConfig

try:
    import wandb
except ImportError:
    wandb = None

class ExperimentManager:
    """Manages files, paths, and metadata for training experiments."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "_dreamer"
        self.base_dir = Path(__file__).resolve().parent
        
        # Scenario-based paths
        scenario_name = cfg.scenario.name
        self.log_dir = self.base_dir / "runs" / scenario_name / self.run_id
        self.ckpt_dir = self.base_dir / "checkpoints" / scenario_name / self.run_id
        self.video_dir = self.ckpt_dir / "videos"
        
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories for logging and checkpoints."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, curriculum):
        """Persist Hydra config and curriculum details to JSON."""
        config_path = self.log_dir / "config.json"
        data = {
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
            "curriculum": {
                "name": curriculum.name,
                "scenario": curriculum.scenario,
                "stages": [asdict(s) for s in curriculum.stages]
            },
            "timestamp": datetime.now().isoformat()
        }
        with open(config_path, "w") as f:
            json.dump(data, f, indent=4)

    def update_manifest(self, curriculum_name):
        """Append run info to a central CSV manifest."""
        manifest_path = self.log_dir.parent.parent / "experiments_manifest.csv"
        file_exists = manifest_path.exists()
        
        with open(manifest_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["RunID", "Date", "Scenario", "Curriculum", "Policy", "Model", "LogDir"])
            
            writer.writerow([
                self.run_id, 
                datetime.now().isoformat(), 
                self.cfg.scenario.name, 
                curriculum_name, 
                "DreamerV3",
                "RSSM",
                str(self.log_dir)
            ])

    def get_stage_log_dir(self, stage_name):
        return self.log_dir / stage_name

    def get_stage_ckpt_dir(self, stage_name):
        path = self.ckpt_dir / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_video_dir(self, stage_name):
        path = self.video_dir / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def init_wandb(self):
        """Initialize Weights & Biases if enabled."""
        if wandb and self.cfg.wandb.enabled:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                group=self.cfg.wandb.group,
                name=self.cfg.wandb.name or self.run_id,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                mode=self.cfg.wandb.mode
            )

    def finish_wandb(self):
        """Finish Weights & Biases run."""
        if wandb and self.cfg.wandb.enabled:
            wandb.finish()
