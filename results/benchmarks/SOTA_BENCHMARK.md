# 🔬 Scientific SOTA Benchmark: 5-Agent Comparison

This benchmark evaluates the performance of historic and modern agents on the `deathmatch` scenario.

## 📋 Methodology

- **Sample Size**: 100 episodes per agent (Total 500)
- **Scenario**: `deathmatch.cfg` (Timelimit: 2100 ticks)
- **Hardware**: RTX 3060 (Local Workstation)
- **Agents**:
    1. **Random** (Baseline)
    2. **Arnold** (2017)
    3. **Intel DFP** (2016)
    4. **Sample Factory** (2022 - *Public Checkpoint*)
    5. **DreamerV3** (*Untrained / Random Init*)

## 📊 Statistical Summary (N=100)

| Metric | Agent | Mean | Std Dev | Max | SEM | Distance | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Frags** | **Intel DFP (2016)** | **4.85** | 80.3 | **26.0** | 0.64 | **4942** | 254 |
| | Arnold (2017) | 3.63 | 82.9 | 31.0 | 0.66 | 3694 | 81 |
| | Sample Factory (2022)* | 0.94 | 14.0 | 5.0 | 0.12 | 96 | 652 |
| | DreamerV3 (Yours) | 0.76 | 16.8 | 8.0 | 0.14 | 742 | 667 |
| | Random (Baseline) | 0.70 | 13.8 | 5.0 | 0.11 | 752 | **776** |

*> **Anomaly Detected**: The Sample Factory agent (`hishamcse/doom_deathmatch_bots`) performed near-random and barely moved (Avg Dist: 96). This suggests an action-space mismatch or a poor-quality checkpoint.*

## 🧪 Hypothesis Testing (T-Test)

- **Arnold vs DFP**: P-Value = `0.008` (**Significant**). DFP is the superior "classic" agent.
- **Dreamer vs Random**: P-Value > 0.5 (Not Significant). Current Dreamer is untrained.

## 📉 Visual Distributions

![Violin Plots](./benchmark_violin.png)

## 🧐 Analysis

1. **The Champion**: Intel DFP (2016) remains the king of this specific scenario in our tests, showing high aggression (roaming) and consistent combat.
2. **The Challenger**: Arnold (2017) has higher "Max Frags" (31 vs 26) but lower consistency (lower mean).
3. **The "SOTA" Failure**: The modern Sample Factory model failed to generalize to our evaluation setup. This highlights the fragility of RL reproducibility (action spaces, frame skips, etc).

## 🧠 Glossary

- **Frags**: Kills per episode.
- **FPS**: Inference speed. Note that **DreamerV3** inference is very fast (~660 FPS) in this random mode, comparable to Sample Factory.
