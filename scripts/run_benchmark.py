import cv2
import random
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Add project root and src to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

from doom_agent.algorithms.dreamer.v3.doom_envs import DoomDreamerEnv
try:
    from scripts.arnold_adapter import ArnoldAdapter
    from scripts.dfp_adapter import DFPAdapter
except (ImportError, ModuleNotFoundError):
    from arnold_adapter import ArnoldAdapter
    from dfp_adapter import DFPAdapter
def run_benchmark(episodes=10, scenario='deathmatch.cfg'):
    print(f"--- 🏁 Benchmarking SOTA Agents: Arnold vs DFP ({episodes} episodes) ---")
    
    # 1. Setup Environment
    env = DoomDreamerEnv(
        scenario=scenario,
        obs_shape=(64, 64, 3), # Consistent shape
        window_visible=False, # Headless benchmarking
        frame_skip=4
    )
    
    results = []
    
    agents = [
        ("Arnold", lambda: ArnoldAdapter(str(root_dir / "external/arnold/pretrained/vizdoom_2017_track2.pth"))),
        ("Intel DFP", lambda: DFPAdapter())
    ]
    
    for agent_name, adapter_factory in agents:
        print(f"\n🚀 Running {agent_name}...")
        try:
            agent = adapter_factory()
            
            for ep in tqdm(range(episodes), desc=f"{agent_name}"):
                obs = env.reset()
                info = {'health': 100, 'ammo': 50, 'frags': 0}
                done = False
                total_reward = 0
                steps = 0
                
                while not done:
                    action_vec = agent.select_action(
                        obs, 
                        health=info.get('health', 100), 
                        ammo=info.get('ammo', 50),
                        frags=info.get('frags', 0)
                    )
                    obs, reward, done, info = env.step_manual(action_vec)
                    total_reward += reward
                    steps += 1
                    
                    if steps > 2100: # Timeout safety (match DFP config)
                        break
                
                results.append({
                    "Agent": agent_name,
                    "Episode": ep + 1,
                    "Total Reward": total_reward,
                    "Frags": info.get('frags', 0),
                    "Final Health": info.get('health', 0),
                    "Steps": steps
                })
                
                if hasattr(agent, "reset"):
                    agent.reset()
                    
        except Exception as e:
            print(f"Error running {agent_name}: {e}")
            import traceback
            traceback.print_exc()

    env.close()
    
    # 2. Save Results
    df = pd.DataFrame(results)
    benchmark_dir = root_dir / "results/benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = benchmark_dir / "sota_comparison_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Raw results saved to {csv_path}")
    
    # 3. Generate Summary Table
    summary = df.groupby("Agent").agg({
        "Total Reward": ["mean", "std"],
        "Frags": ["mean", "max"],
        "Steps": "mean"
    }).round(2)
    
    print("\n--- Summary Table ---")
    print(summary)
    
    summary_path = benchmark_dir / "sota_summary.md"
    with open(summary_path, "w") as f:
        f.write("# SOTA Benchmark Summary\n\n")
        f.write("```csv\n")
        f.write(summary.to_csv())
        f.write("```\n")
   
    # 4. Generate Charts
    print("\n📈 Generating charts...")
    sns.set_theme(style="darkgrid")
    
    # Fig 1: Reward Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Agent", y="Total Reward", data=df, palette="viridis")
    plt.title(f"Cumulative Reward Distribution ({scenario})", fontsize=14, fontweight='bold')
    plt.savefig(benchmark_dir / "benchmark_rewards.png", dpi=150)
    plt.close()
    
    # Fig 2: Frags Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Agent", y="Frags", data=df, palette="magma", errorbar="sd")
    plt.title(f"Average Frags per Episode ({scenario})", fontsize=14, fontweight='bold')
    plt.savefig(benchmark_dir / "benchmark_frags.png", dpi=150)
    plt.close()
    
    print(f"✅ Benchmark complete! View results in {benchmark_dir}")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per agent")
    args = parser.parse_args()
    
    run_benchmark(episodes=args.episodes)
