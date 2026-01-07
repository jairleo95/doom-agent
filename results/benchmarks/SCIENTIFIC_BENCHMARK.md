# 🔬 Scientific Benchmark: Arnold vs DFP

**Sample Size**: 10 episodes per agent

## 📊 Statistical Summary
```csv
,Total Reward,Total Reward,Frags,Frags,Frags,Distance Traveled,FPS
,mean,std,mean,max,sem,mean,mean
Agent,,,,,,,
Arnold (2017),75.1,97.43,6.2,23.0,2.46,4688.14,74.06
DreamerV3 (Yours),10.1,19.04,0.9,5.0,0.5,734.54,859.76
Intel DFP (2016),21.3,24.73,1.8,6.0,0.63,3740.23,261.26
Random (Baseline),11.8,15.11,1.1,4.0,0.41,705.58,1082.99
Sample Factory (2022),12.0,,1.0,1.0,,279.66,260.17
```


## 🧠 Metrics Analysis
- **Frags**: Kills per episode. Primary measure of combat effectiveness.
- **Distance Traveled**: Proxy for exploration and non-camping behavior.
- **Health Loss**: Damage taken. Lower is better (defensive skill).
- **FPS**: Inference speed on current hardware.

## 📈 Distributions
![Violin Plot](./benchmark_violin.png)
