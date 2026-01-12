import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def load_metrics(path):
    metrics_file = Path(path) / "metrics.json"
    if not metrics_file.exists():
        # Try finding it in a subfolder (sometimes hydra adds a timestamp or similar)
        json_files = list(Path(path).rglob("metrics.json"))
        if not json_files:
            return None
        metrics_file = json_files[0]
        
    with open(metrics_file, "r") as f:
        return json.load(f)

def analyze_ablations(base_dir="results/ablations"):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} not found.")
        return

    ablations = [d.name for d in base_path.iterdir() if d.is_dir()]
    if not ablations:
        print("No ablation directories found.")
        return

    results = {}
    for name in ablations:
        m = load_metrics(base_path / name)
        if m:
            results[name] = m
        else:
            print(f"Warning: No metrics found for {name}")

    if "baseline" not in results:
        print("Error: 'baseline' run not found. Cannot perform T-tests.")
        return

    baseline_rewards = results["baseline"].get("eval_mean_rewards", [])
    if not baseline_rewards:
        # Fallback to episode rewards if eval not available
        baseline_rewards = results["baseline"].get("episode_rewards", [])

    print("# DreamerV3 Ablation Study Comparison\n")
    print("| Variant | Mean Reward | Mean Frags | Δ Baseline | P-Value (T-Test) |")
    print("| :--- | :--- | :--- | :--- | :--- |")

    for name, m in results.items():
        rewards = m.get("eval_mean_rewards", [])
        if not rewards: rewards = m.get("episode_rewards", [])
        
        frags = m.get("frags", [0])
        
        mean_reward = np.mean(rewards) if rewards else 0
        mean_frags = np.mean(frags) if frags else 0
        
        # T-Test vs Baseline
        if name != "baseline":
            t_stat, p_val = stats.ttest_ind(baseline_rewards, rewards, equal_var=False)
            delta = mean_reward - np.mean(baseline_rewards)
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        else:
            delta = 0
            p_str = "-"

        print(f"| **{name}** | {mean_reward:.2f} | {mean_frags:.2f} | {delta:+.2f} | {p_str} |")

    # Generate Comparison Plot
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    means = [np.mean(results[n].get("eval_mean_rewards", results[n].get("episode_rewards", [0]))) for n in names]
    stds = [np.std(results[n].get("eval_mean_rewards", results[n].get("episode_rewards", [0]))) for n in names]

    plt.bar(names, means, yerr=stds, capsize=5, color='skyblue', edgecolor='navy')
    plt.ylabel("Mean Reward (last 100 eps or eval)")
    plt.title("DreamerV3 Ablation Performance Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = base_path / "ablation_comparison.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")

if __name__ == "__main__":
    analyze_ablations()
