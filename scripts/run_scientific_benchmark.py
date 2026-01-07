import cv2
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from scipy import stats

# Add project root and src to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

from doom_agent.algorithms.dreamer.v3.doom_envs import DoomDreamerEnv
# Imports with fallback
try:
    from scripts.arnold_adapter import ArnoldAdapter
    from scripts.dfp_adapter import DFPAdapter
    from scripts.random_adapter import RandomAdapter
    from scripts.dreamer_adapter import DreamerV3Adapter
    from scripts.sf_adapter import SampleFactoryAdapter
except (ImportError, ModuleNotFoundError):
    from arnold_adapter import ArnoldAdapter
    from dfp_adapter import DFPAdapter
    from random_adapter import RandomAdapter
    from dreamer_adapter import DreamerV3Adapter
    from sf_adapter import SampleFactoryAdapter

class ScientificBenchmark:
    def __init__(self, episodes=30, scenario='deathmatch.cfg'):
        self.episodes = episodes
        self.scenario = scenario
        self.results_dir = root_dir / "results/benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.env = DoomDreamerEnv(
            scenario=scenario,
            obs_shape=(64, 64, 3),
            window_visible=False,
            frame_skip=4
        )
        
    def run(self):
        print(f"--- 🔬 Running Scientific Benchmark ({self.episodes} samples/agent) ---")
        
        agents = [
            ("Random (Baseline)", lambda: RandomAdapter()),
            ("Arnold (2017)", lambda: ArnoldAdapter(str(root_dir / "external/arnold/pretrained/vizdoom_2017_track2.pth"))),
            ("Intel DFP (2016)", lambda: DFPAdapter()),
            ("Sample Factory (2022)", lambda: SampleFactoryAdapter(str(root_dir / "external/sample_factory_model"))),
            ("DreamerV3 (Yours)", lambda: DreamerV3Adapter(str(root_dir / "results/latest_model.ckpt")))
        ]
        
        all_metrics = []
        
        for agent_name, factory in agents:
            print(f"\n🧪 Evaluating {agent_name}...")
            try:
                agent = factory()
                
                for ep in tqdm(range(self.episodes), desc=agent_name):
                    metrics = self._run_episode(agent, ep, agent_name)
                    all_metrics.append(metrics)
                    
                    if hasattr(agent, "reset"):
                        agent.reset()
                        
            except Exception as e:
                print(f"❌ Failed to run {agent_name}: {e}")
                import traceback
                traceback.print_exc()

        self.env.close()
        
        # Process Data
        df = pd.DataFrame(all_metrics)
        raw_csv = self.results_dir / "scientific_benchmark_raw.csv"
        df.to_csv(raw_csv, index=False)
        print(f"\n💾 Raw data saved to {raw_csv}")
        
        self._generate_report(df)
        self._generate_plots(df)
        
    def _run_episode(self, agent, ep_num, agent_name):
        obs = self.env.reset()
        done = False
        info = {'health': 100, 'ammo': 50, 'frags': 0, 'pos_x': 0, 'pos_y': 0}
        
        # Metrics trackers
        start_time = time.time()
        start_ammo = 50
        max_health = 100
        
        total_reward = 0
        steps = 0
        loss_health_accum = 0
        ammo_consumed = 0
        
        # Movement tracking
        positions = []
        
        while not done:
            action_vec = agent.select_action(
                obs, 
                health=info.get('health', 100), 
                ammo=info.get('ammo', 50),
                frags=info.get('frags', 0)
            )
            
            # Count ammo usage (approximation: if we fired and ammo decreased)
            prev_ammo = info.get('ammo', 50)
            prev_health = info.get('health', 100)
            
            obs, reward, done, info = self.env.step_manual(action_vec)
            
            # Delta Tracking
            curr_ammo = info.get('ammo', 50)
            curr_health = info.get('health', 100)
            
            if curr_ammo < prev_ammo:
                ammo_consumed += (prev_ammo - curr_ammo)
            
            if curr_health < prev_health:
                loss_health_accum += (prev_health - curr_health)
            
            positions.append((info.get('pos_x', 0), info.get('pos_y', 0)))
            
            total_reward += reward
            steps += 1
            
            if steps > 2100: break # Safety limit

        duration = time.time() - start_time
        
        # Calculate Displacement
        dist_traveled = 0
        for i in range(1, len(positions)):
            p1 = positions[i-1]
            p2 = positions[i]
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if dist < 50: # Filter teleports/respawns roughly
                dist_traveled += dist

        return {
            "Agent": agent_name,
            "Episode": ep_num + 1,
            "Total Reward": total_reward,
            "Frags": info.get('frags', 0),
            "Survival Steps": steps,
            "Health Loss": loss_health_accum,
            "Ammo Consumed": ammo_consumed,
            "Distance Traveled": dist_traveled,
            "FPS": steps / (duration + 1e-6)
        }

    def _generate_report(self, df):
        summary_path = self.results_dir / "SCIENTIFIC_BENCHMARK.md"
        
        stats_df = df.groupby("Agent").agg({
            "Total Reward": ["mean", "std"],
            "Frags": ["mean", "max", "sem"], # sem = Standard Error of Mean
            "Distance Traveled": "mean",
            "FPS": "mean"
        }).round(2)
        
        # T-Test Calculation
        agents = df["Agent"].unique()
        significance_msg = ""
        if len(agents) == 2:
            a1 = df[df["Agent"] == agents[0]]["Frags"]
            a2 = df[df["Agent"] == agents[1]]["Frags"]
            t_stat, p_val = stats.ttest_ind(a1, a2, equal_var=False)
            
            sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
            significance_msg = f"\n\n### Statistical Significance (Welch's t-test)\n" \
                               f"- **Comparing Frags**: {agents[0]} vs {agents[1]}\n" \
                               f"- **P-Value**: {p_val:.4e}\n" \
                               f"- **Result**: Difference is **{sig}** (alpha=0.05)"

        with open(summary_path, "w") as f:
            f.write("# 🔬 Scientific Benchmark: Arnold vs DFP\n\n")
            f.write(f"**Sample Size**: {self.episodes} episodes per agent\n\n")
            f.write("## 📊 Statistical Summary\n")
            f.write("```csv\n")
            f.write(stats_df.to_csv())
            f.write("```\n")
            f.write(significance_msg)
            f.write("\n\n## 🧠 Metrics Analysis\n")
            f.write("- **Frags**: Kills per episode. Primary measure of combat effectiveness.\n")
            f.write("- **Distance Traveled**: Proxy for exploration and non-camping behavior.\n")
            f.write("- **Health Loss**: Damage taken. Lower is better (defensive skill).\n")
            f.write("- **FPS**: Inference speed on current hardware.\n\n")
            f.write("## 📈 Distributions\n")
            f.write("![Violin Plot](./benchmark_violin.png)\n")
        
        print(f"📄 Report generated: {summary_path}")

    def _generate_plots(self, df):
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        
        # 1. Multi-metric Violin Plots
        metrics = ["Total Reward", "Frags", "Distance Traveled", "Ammo Consumed"]
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            sns.violinplot(x="Agent", y=metric, data=df, ax=ax, inner="quart", palette="muted")
            sns.stripplot(x="Agent", y=metric, data=df, ax=ax, color="black", alpha=0.3, jitter=True)
            ax.set_title(metric, fontweight='bold')
            ax.set_xlabel("")
            
        plt.suptitle(f"Metric Distributions (N={self.episodes})", fontsize=18, y=0.98)
        plt.tight_layout()
        plt.savefig(self.results_dir / "benchmark_violin.png", dpi=200)
        plt.close()
        
        # 2. Frag vs Survival Scatter
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x="Survival Steps", y="Frags", hue="Agent", style="Agent", s=100, alpha=0.7, data=df, palette="deep")
        plt.title("Combat Efficiency: Frags vs Survival Duration", fontweight='bold')
        plt.savefig(self.results_dir / "benchmark_scatter.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="Sample size per agent")
    args = parser.parse_args()
    
    bench = ScientificBenchmark(episodes=args.episodes)
    bench.run()
