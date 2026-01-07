import wandb
import argparse
import os
from pathlib import Path
import shutil

def download_model(run_id, project="jairleo95/doom-dreamer", entity=None):
    # Initialize wandb API
    api = wandb.Api()
    
    # Artifact name format used in train.py: f"{run_id}_best_model"
    # Full path: entity/project/artifact_name:alias
    artifact_path = f"{project}/{run_id}_best_model:latest"
    if entity:
        artifact_path = f"{entity}/{artifact_path}"
        
    print(f"Downloading artifact: {artifact_path}")
    
    try:
        artifact = api.artifact(artifact_path)
        download_dir = Path(f"downloads/{run_id}")
        
        # Force replacement by deleting existing directory
        if download_dir.exists():
            print(f"Directory {download_dir} already exists. Cleaning up for fresh download...")
            shutil.rmtree(download_dir)
            
        download_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting download to {download_dir}...")
        artifact_dir = artifact.download(root=str(download_dir))
        print(f"Successfully downloaded to: {artifact_dir}")
        
        # Look for the .pt file
        pt_files = list(Path(artifact_dir).glob("*.pt"))
        if pt_files:
            return pt_files[0]
        else:
            print("No .pt file found in artifact.")
            return None
            
    except Exception as e:
        print(f"Error downloading artifact: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="W&B Run ID (e.g. 20260101-054831_dreamer)")
    parser.add_argument("--project", type=str, default="doom-dreamer", help="W&B Project name")
    parser.add_argument("--entity", type=str, default=None, help="W&B Entity/Username")
    
    args = parser.parse_args()
    
    model_path = download_model(args.run_id, args.project, args.entity)
    if model_path:
        print(f"\nTo visualize, run:")
        print(f"export PYTHONPATH=$PYTHONPATH:$(pwd)/src")
        print(f"python src/doom_agent/algorithms/dreamer/v3/visualize.py --path {model_path} --scenario deathmatch --fps 35")
