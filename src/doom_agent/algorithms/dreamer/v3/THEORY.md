# 🧠 Scientific Foundation: World Models & Imitation Learning in VizDoom

This document explains the theoretical underpinnings of the agents used in this research, specifically focusing on the transition from specialized architectures to general-purpose World Models.

## 1. Dreamer V3: Mastering Diverse Domains

[Hafner et al., 2023]

DreamerV3 builds upon the Model-Based Reinforcement Learning (MBRL) paradigm. Unlike Model-Free agents (PPO, DQN) that learn a direct mapping from observations to actions, Dreamer learns a **World Model** to simulate the environment.

### Recurrent State-Space Model (RSSM)

The agent maintains a latent state $(h_t, z_t)$ where:

- **$h_t$ (Deterministic)**: A GRU-based history representation.
- **$z_t$ (Stochastic)**: Categorical variables (32x32) that represent discrete concepts. Discretization prevents the world model from collapsing and improves robustness to pixel-level noise.

### Components

1. **Encoder/Decoder**: Maps pixels to latents and back. Uses **SymLog** scaling to handle rewards and observations of varying magnitudes.
2. **Dynamics (Prior)**: Learns to predict $p(z_{t+1}|h_t, a_t)$ without seeing the next frame.
3. **Representations (Posterior)**: Learns $q(z_t|h_t, x_t)$ by seeing the actual observation.
4. **Behavior Learning**: The agent trains in **Imagination** for ~15 steps into the future, using its internal world model to optimize an Actor-Critic pair.

---

## 2. Historical SOTA Baselines

### Arnold (2017)

[Lample & Chaplot, "Playing FPS Games with Deep Reinforcement Learning"]

- **Architecture**: DRQN (Deep Recurrent Q-Network).
- **Key Innovation**: Augmented the agent with secondary game features (e.g., enemy detection, item presence) and used separate networks for navigation and combat.
- **Performance**: Won the 2017 VizDoom AI Competition.

### Direct Future Prediction (DFP, 2016)

[Dosovitskiy & Koltun, "Learning to Act by Predicting the Future"]

- **Architecture**: Specialized Multi-Objective branch.
- **Key Innovation**: Bypassed standard Reinforcement Learning (Bellman equations) in favor of supervised-style prediction of "future measurements" (Health, Ammo, Frags) at multiple time offsets.
- **Performance**: Dominant baseline in early VizDoom research.

### Sample Factory (2020/2022)

[Petrenko et al., "High-throughput 3D Control with Parallel Reinforcement Learning"]

- **Architecture**: High-performance APPO (Asynchronous Proximal Policy Optimization).
- **Contribution**: Proved that specialized, high-throughput model-free architectures can reach extreme performance by processing billions of frames per day.

---

## 3. Imitation Learning: Behavior Cloning (BC)

Directly applying RL in sparse reward environments like VizDoom Deathmatch often leads to slow convergence (the "Cold Start" problem).

### Behavioral Cloning Pipeline

$L_{BC} = \mathbb{E}_{(s,a) \sim \mathcal{D}_{expert}} [-\log \pi(a|s)]$

We utilize **Arnold** to harvest successful trajectories. By initializing the DreamerV3 Actor network via BC, the agent starts its interaction phase already knowing how to:

1. Search for enemies.
2. Maintain aim stability.
3. Prioritize survival (strafing).

This "Jumpstart" significantly reduces the number of environment interactions needed to reach champion-level performance.

---

## 4. Academic References

- **DreamerV3**: Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models. *arXiv preprint arXiv:2301.04104*.
- **Arnold**: Lample, G., & Chaplot, D. S. (2017). Playing FPS games with deep reinforcement learning. In *Thirty-First AAAI Conference on Artificial Intelligence*.
- **DFP**: Dosovitskiy, A., & Koltun, V. (2016). Learning to act by predicting the future. *arXiv preprint arXiv:1611.01779*.
- **Sample Factory**: Petrenko, A., Huang, Z., Kumar, T., Sukhatme, G., & Koltun, V. (2020). Sample factory: Egg-centric high-throughput 3D control. *International Conference on Machine Learning*.
- **ViZDoom**: Wydmuch, M., Kempka, M., & Jaśkowski, G. (2018). ViZDoom: A Doom-based AI research platform for visual reinforcement learning. *IEEE Transactions on Games*.
