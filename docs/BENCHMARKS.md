# 🔬 Benchmarking Científico y Comparativa SOTA

Este documento describe la suite de benchmarking, la metodología de evaluación experimental y los agentes de referencia integrados en [doom-agent](file:///home/darkstar/Workspace/ai/rl/doom-agent/README.md).

---

## 1. Agentes y Modelos de Referencia (SOTA)

Para validar el rendimiento de las arquitecturas propuestas de forma reproducible e imparcial, el repositorio integra implementaciones oficiales de los hitos históricos en ViZDoom:

| Agente | Año | Paradigma | Innovación Clave | Ubicación del Adaptador |
|---|---|---|---|---|
| **Random Baseline** | N/A | Heurístico | Selección aleatoria uniforme de acciones para control base | [random_adapter.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/scripts/random_adapter.py) |
| **Intel DFP** | 2016 | Direct Future Prediction | Predicción supervisada de métricas futuras (Salud, Balas, Frags) sin Bellman | [dfp_adapter.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/scripts/dfp_adapter.py) / [external/dfp](file:///home/darkstar/Workspace/ai/rl/doom-agent/external/dfp) |
| **Arnold** | 2017 | Model-Free (DRQN) | Campeón ViZDoom 2017. Redes desacopladas para navegación y combate | [arnold_adapter.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/scripts/arnold_adapter.py) / [external/arnold](file:///home/darkstar/Workspace/ai/rl/doom-agent/external/arnold) |
| **Sample Factory** | 2022 | Model-Free (APPO) | PPO asíncrono optimizado con alto paralelismo y throughput masivo | [sf_adapter.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/scripts/sf_adapter.py) |
| **DreamerV3** | 2023 | Model-Based RL (RSSM) | Imaginación en espacio latente discreto, escalado SymLog y mirror learning | [DreamerV3Agent](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/agent.py#L61) |

---

## 2. Metodología del Benchmark Científico

El script [run_scientific_benchmark.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/scripts/run_scientific_benchmark.py) orquesta la evaluación comparativa mediante la clase [ScientificBenchmark](file:///home/darkstar/Workspace/ai/rl/doom-agent/scripts/run_scientific_benchmark.py#L32):

- **Escenario**: `deathmatch.cfg` (combate abierto contra bots de ViZDoom bajo semillas fijas).
- **Muestreo**: 30 episodios por agente para significancia estadística.
- **Métricas Registradas**:
  - **Frags**: Bajas enemigas confirmadas por episodio.
  - **Deaths**: Muertes sufridas por el agente.
  - **K/D Ratio**: Ratio de eficacia en combate ($\frac{\text{Frags}}{\max(1, \text{Deaths})}$).
  - **Survival Time**: Tiempo de vida promedio en ticks antes de ser eliminado.
  - **Net Reward**: Recompensa acumulada según las políticas de reward shaping.
  - **Throughput (FPS)**: Rendimiento computacional de inferencia.

### Ejecución del Benchmark

```bash
# Asegurar el path del proyecto
export PYTHONPATH="$(pwd)/src:$(pwd)"

# Ejecutar la suite completa de evaluación científica
python scripts/run_scientific_benchmark.py
```

Los resultados se exportan automáticamente a:
- `results/benchmarks/scientific_benchmark_results.csv` (Métricas crudas)
- `results/benchmarks/benchmark_comparison.png` (Boxplots y diagramas de barras con intervalos de confianza)

---

## 3. Validador Interactivo de Reward Shaping

Antes de lanzar entrenamientos prolongados en GPUs o clústeres cloud, es crucial verificar el comportamiento de los incentivos (hambre de combate, penalización de daño, gasto de munición). Para ello se utiliza [test_reward_shaping.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/scripts/test_reward_shaping.py).

### Modos de Validación

1. **Modo Manual (Humano)**:
   Permite controlar al jugador con teclado (`W/S/A/D/Q/E/Space`) para experimentar en tiempo real el valor numérico de las recompensas generadas:
   ```bash
   python scripts/test_reward_shaping.py --manual
   ```

2. **Modo Espectador AI (DreamerV3 Checkpoint)**:
   Supervisa el comportamiento en vivo de un agente entrenado bajo un identificador de corrida:
   ```bash
   python scripts/test_reward_shaping.py --agent_run <RUN_ID>
   ```

3. **Modo Arnold ("Boss Mode")**:
   Carga el modelo preentrenado del campeón mundial 2017 para observar cómo sus decisiones activan las recompensas diseñadas:
   ```bash
   # Configuración inicial de Arnold (descarga de pesos y dependencias)
   bash scripts/setup_arnold.sh

   # Ejecución en modo espectador
   python scripts/test_reward_shaping.py --arnold
   ```

4. **Modo Intel DFP**:
   Evalúa el agente basado en predicción directa de medidas futuras:
   ```bash
   # Configuración y parches de compatibilidad TF2
   bash scripts/setup_dfp.sh

   # Ejecución en modo espectador
   python scripts/test_reward_shaping.py --dfp
   ```

---

## 4. Pruebas de Ablación

El script [compare_ablations.py](file:///home/darkstar/Workspace/ai/rl/doom-agent/scripts/compare_ablations.py) y la suite [run_ablation_suite.sh](file:///home/darkstar/Workspace/ai/rl/doom-agent/scripts/run_ablation_suite.sh) permiten medir el impacto aislado de componentes clave:
- Impacto de **Behavior Cloning** (Pre-entrenamiento offline vs. RL puro desde cero).
- Impacto de **Mirror Learning** (Aumentación de simetría horizontal).
- Impacto de **Curriculum Learning** (Entrenamiento escalonado Skill 2 $\rightarrow$ Skill 5 vs. entrenamiento directo en Skill 5).
