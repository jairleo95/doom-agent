import wandb
import pandas as pd
import os
from pathlib import Path
import json
from plot_results import plot_run_metrics

def export_wandb_summary(run_id, project="doom-agent", entity=None):
    api = wandb.Api()
    
    # Path: entity/project/run_id
    run_path = f"{project}/{run_id}"
    if entity:
        run_path = f"{entity}/{run_path}"
        
    print(f"Fetching run data: {run_path}")
    
    try:
        # Try direct ID lookup first
        try:
            run = api.run(run_path)
            print(f"Found run by ID: {run_id}")
        except Exception:
            # Fallback: Search by name
            print(f"ID lookup failed, searching for run with name: {run_id}...")
            full_project_path = f"{project}"
            if entity: full_project_path = f"{entity}/{full_project_path}"
            
            runs = api.runs(full_project_path, filters={"display_name": run_id})
            if len(runs) > 0:
                run = runs[0]
                print(f"Found run by name! ID is: {run.id}")
            else:
                raise Exception(f"Run name '{run_id}' not found in project '{full_project_path}'")
        
        # 1. Save Summary (Final metrics)
        summary = {k: v for k, v in run.summary.items() if not k.startswith('_')}
        
        # Ensure stage_name is preserved
        if "stage_name" in run.summary:
            summary["stage_name"] = run.summary["stage_name"]
        elif "stage" in run.config:
            summary["stage_name"] = run.config["stage"]
        else:
            # Fallback for older runs or missing keys
            summary["stage_name"] = "Skill 4" # Default for current experiments
        
        # 2. Save Config
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        
        # Create output directory
        out_dir = Path(f"results/runs/{run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        # 3. Export History (Time series)
        print(f"Downloading history (Sampling {args.samples} points)...")
        
        if args.full:
            print("Full export requested. Using scan_history (this may be VERY slow)...")
            history_data = []
            for row in run.scan_history(keys=["charts/episode_reward", "gameplay/frags", "charts/fps", "_step"]):
                history_data.append(row)
            history_df = pd.DataFrame(history_data)
        else:
            # Fastest way: Sampled history
            history_df = run.history(
                samples=args.samples, 
                keys=[
                    "charts/episode_reward", "train/eval_mean_reward",
                    "gameplay/frags", "gameplay/health_remaining", "gameplay/ammo_consumed",
                    "charts/fps", "charts/episode_length",
                    "train/kl_free", "train/dyn_scale", "train/rep_scale",
                    "_step", "_runtime"
                ]
            )
        
        if history_df is not None and not history_df.empty:
            history_df.to_csv(out_dir / "history.csv", index=False)
            print(f"Exported {len(history_df)} history samples.")
            
            # Generate local plots
            plot_paths = []
            try:
                plot_paths = plot_run_metrics(run_id)
            except Exception as e:
                print(f"Warning: Could not generate plot: {e}")
        else:
            print("Warning: No history data found for specified keys.")
            plot_paths = []
        
        print(f"Successfully exported data to {out_dir}")
        return summary, out_dir, plot_paths
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, []

def update_results_md(run_id, summary, out_dir, plot_paths=[]):
    results_path = Path("RESULTS.md")
    
    # Header if file is new
    if not results_path.exists():
        with open(results_path, "w") as f:
            f.write("# Training Progress & Results\n\n")
            f.write("This file tracks the best performing runs and their key metrics.\n\n")
    
    # Append latest results
    with open(results_path, "a") as f:
        f.write(f"## Run: {run_id}\n")
        f.write(f"- **Final Reward**: {summary.get('charts/episode_reward', 'N/A'):.2f}\n")
        f.write(f"- **Total Frags**: {summary.get('gameplay/frags', 'N/A')}\n")
        f.write(f"- **Peak FPS**: {summary.get('charts/fps', 'N/A'):.2f}\n")
        
        # Format runtime from seconds
        runtime_sec = summary.get('_runtime', 0)
        runtime_str = f"{runtime_sec/3600:.1f}h" if runtime_sec < 86400 else f"{runtime_sec/86400:.1f}d"
        f.write(f"- **Duration**: {runtime_str}\n")
        
        f.write(f"- **Stage**: {summary.get('stage_name', 'Skill 4')}\n")
        
        if plot_paths:
            f.write("- **Gráficas Detalladas**:\n")
            for p in plot_paths:
                name = p.stem.replace("plot_", "").capitalize()
                f.write(f"  - [{name}](file://{p.absolute()})\n")
        
        f.write(f"- **Data Location**: `results/runs/{run_id}/`\n\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--project", type=str, default="doom-agent")
    parser.add_argument("--samples", type=int, default=1000, help="Number of points to sample for history")
    parser.add_argument("--full", action="store_true", help="Download every single step (slow)")
    args = parser.parse_args()
    
    summary, out_dir, plot_paths = export_wandb_summary(args.run_id, args.project)
    if summary:
        update_results_md(args.run_id, summary, out_dir, plot_paths)
        print("RESULTS.md updated.")
