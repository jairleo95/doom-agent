# 🔬 Scientific Benchmark: Arnold vs DFP

**Sample Size**: 100 episodes per agent

## 📊 Statistical Summary
```csv
,Total Reward,Total Reward,Frags,Frags,Frags,Distance Traveled,FPS
,mean,std,mean,max,sem,mean,mean
Agent,,,,,,,
Arnold (2017),41.44,69.85,3.39,22.0,0.56,3586.69,67.16
Intel DFP (2016),69.79,82.51,5.69,22.0,0.66,4921.97,250.96
```


### Statistical Significance (Welch's t-test)
- **Comparing Frags**: Arnold (2017) vs Intel DFP (2016)
- **P-Value**: 8.3235e-03
- **Result**: Difference is **SIGNIFICANT** (alpha=0.05)

## 🧠 Metrics Analysis
- **Frags**: Kills per episode. Primary measure of combat effectiveness.
- **Distance Traveled**: Proxy for exploration and non-camping behavior.
- **Health Loss**: Damage taken. Lower is better (defensive skill).
- **FPS**: Inference speed on current hardware.

## 📈 Distributions
![Violin Plot](./benchmark_violin.png)
