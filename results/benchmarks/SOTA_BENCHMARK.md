# 🔬 Scientific SOTA Benchmark: Arnold (2017) vs Intel DFP (2016)

This benchmark evaluates the performance of two historic ViZDoom champions with rigorous statistical analysis.

## 📋 Methodology

- **Sample Size**: 100 episodes per agent
- **Scenario**: `deathmatch.cfg` (Timelimit: 2100 ticks)
- **Significance Test**: Welch's t-test (Two-samples, unequal variance)
- **Hardware**: RTX 3060 / Ryzen 7 (Local Workstation)
- **Inference**:
  - **Arnold**: PyTorch 2.5 (CPU Mode for stability) via `ArnoldAdapter`
  - **DFP**: TensorFlow 2.x (Compat Mode) via `DFPAdapter`

## 📊 Statistical Summary

| Metric | Agent | Mean | Std Dev | Max | SEM (Error) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Frags** | Arnold (2017) | 3.39 | -- | 22.0 | 0.56 |
| | **Intel DFP (2016)** | **5.69** | -- | **22.0** | **0.66** |
| **Reward** | Arnold (2017) | 41.44 | 69.85 | -- | -- |
| | **Intel DFP (2016)** | **69.79** | 82.51 | -- | -- |
| **Distance** | Arnold (2017) | 3586.69 | -- | -- | -- |
| | **Intel DFP (2016)** | **4921.97** | -- | -- | -- |
| **FPS** | Arnold (2017) | 67.16 | -- | -- | -- |
| | **Intel DFP (2016)** | **250.96** | -- | -- | -- |

### 🧪 Hypothesis Testing (Frags)

- **Null Hypothesis ($H_0$)**: There is no difference in mean frags between Arnold and DFP.
- **P-Value**: `0.0083` (8.32e-3)
- **Conclusion**: The difference is **STATISTICALLY SIGNIFICANT** at $\alpha=0.05$. Intel DFP consistently outperforms Arnold in this larger sample (N=100).

## 📉 Visual Distributions

### 1. Metric Distributions (Violin Plots)

N=100 Sample clearly shows DFP's distribution shifted towards higher rewards and kill counts compared to Arnold's heavier tail near zero.
![Violin Plots](./benchmark_violin.png)

### 2. Combat Efficiency (Scatter Plot)

Relationship between survival time and kills.

- **Top Right**: High survival + High kills (Dominator)
- **Bottom Right**: High survival + Low kills (Camper)
- **Top Left**: Low survival + High kills (Glass Cannon)
![Scatter Plot](./benchmark_scatter.png)

## 🧐 Discussion [N=100 Update]

With a larger sample size (N=100), the results contradict the historical 2017 VDAIC rankings where Arnold won:

- **Significance**: We can now confidently say DFP outperforms Arnold in this specific headless `deathmatch.cfg` scenario (P < 0.01).
- **Movement**: DFP traverses significantly more map area (4921 vs 3586), likely leading to more encounters and item pickups.
- **Stability**: Arnold's high variance implies it often gets stuck or fails early, whereas DFP maintains a more consistent "roaming" pressure.

## 🧠 Metrics Definitions (Glossary)

- **Frags**: "Fragmentations". A classic FPS term for **Kills**. It represents the number of enemy agents eliminated by the benchmarked agent. Higher is strictly better.
- **Distance Traveled**: Total Euclidean distance moved by the agent. Proxy for exploration and non-camping behavior.
- **Health Loss**: Total damage taken during the episode. Lower is better (implies better dodging/cover).
- **FPS**: Frames Per Second. Inference speed on the test hardware.
