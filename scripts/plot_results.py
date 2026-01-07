import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
from pathlib import Path

def plot_run_metrics(run_id, results_dir="results/runs"):
    run_dir = Path(results_dir) / run_id
    history_path = run_dir / "history.csv"
    
    if not history_path.exists():
        print(f"Error: History file not found at {history_path}")
        return
    
    # Load data
    df = pd.read_csv(history_path)
    if df.empty:
        print("Error: History file is empty.")
        return

    # Clean column names (remove prefixes for better labeling)
    df.columns = [c.split('/')[-1] if '/' in str(c) else c for c in df.columns]

    # Setup aesthetic
    sns.set_theme(style="darkgrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    
    # Delete legacy plot to avoid confusion
    legacy_plot = run_dir / "progress_plot.png"
    if legacy_plot.exists():
        legacy_plot.unlink()
    
    # Identify key metrics
    groups = {
        "Rewards": ["episode_reward", "eval_mean_reward"],
        "Gameplay": ["frags", "health_remaining", "ammo_consumed"],
        "Performance": ["fps", "episode_length"],
        "World Model": ["kl_free", "dyn_scale", "rep_scale"]
    }
    
    # Try to get stage name for context
    stage_name = "Training"
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
                stage_name = summary_data.get("stage_name", summary_data.get("stage", "Training"))
        except: pass

    # Get duration if available
    duration_str = ""
    if "_runtime" in df.columns:
        total_sec = df["_runtime"].iloc[-1]
        if total_sec > 86400:
            duration_str = f"Time: {total_sec/86400:.1f}d"
        else:
            duration_str = f"Time: {total_sec/3600:.1f}h"
    elif "runtime" in df.columns: # Sometimes cleaned
        total_sec = df["runtime"].iloc[-1]
        duration_str = f"Time: {total_sec/3600:.1f}h"

    # Filter available groups
    active_groups = {}
    for g_name, g_metrics in groups.items():
        avail = [m for m in g_metrics if m in df.columns]
        if avail:
            active_groups[g_name] = avail

    if not active_groups:
        print("No plotable metrics found in CSV.")
        return

    # Create plots for each group
    x_axis_data = None
    x_label = "Episodes"
    
    if "_step" in df.columns: 
        x_axis_data = df["_step"] / 1e6
        x_label = "Steps (Millions)"
    elif "step" in df.columns: 
        x_axis_data = df["step"] / 1e6
        x_label = "Steps (Millions)"
    else: 
        x_axis_data = df.index
    
    saved_paths = []
    
    for g_name, g_metrics in active_groups.items():
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="darkgrid") # Reset for each figure
        
        for m in g_metrics:
            # Add smoothing for better visibility
            if len(df) > 10:
                window = max(2, len(df)//20)
                smoothed = df[m].rolling(window=window, min_periods=1).mean()
                sns.lineplot(x=x_axis_data, y=smoothed, label=f"{m.capitalize()} (Smooth)")
                # Original data with lower alpha
                sns.lineplot(x=x_axis_data, y=df[m], alpha=0.2, legend=None)
            else:
                sns.lineplot(x=x_axis_data, y=df[m], label=m.capitalize())
        
        title_full = f"{g_name} - {stage_name} ({run_id})"
        if duration_str: title_full += f" | {duration_str}"
        
        plt.title(title_full, fontsize=14, fontweight='bold')
        plt.ylabel("Value")
        plt.xlabel(x_label)
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        # Save individual group plot
        file_name = f"plot_{g_name.lower().replace(' ', '_')}.png"
        plot_path = run_dir / file_name
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"Group plot saved: {plot_path}")
        saved_paths.append(plot_path)
    
    return saved_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="W&B Run ID/Name")
    args = parser.parse_args()
    
    plot_run_metrics(args.run_id)
