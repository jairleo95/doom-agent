# DreamerV3: Dominando Dominios Diversos con Modelos del Mundo

DreamerV3 es un algoritmo de Reinforcement Learning (RL) basado en model-based learning que destaca por su capacidad de entrenar agentes en una enorme variedad de dominios (desde control continuo hasta tareas visuales complejas como Doom) con un único conjunto de hiperparámetros fijos.

## Arquitectura Principal: El Modelo del Mundo (World Model)

A diferencia de PPO, que aprende directamente de la experiencia, DreamerV3 aprende a "soñar". Construye un modelo interno del entorno compuesto por:

### 1. Percepción y Codificación
- **Encoder (CNN)**: Transforma las imágenes de entrada (e.g., frames de Doom) en representaciones de características compactas.
- **SymLog Transformation**: Aplica `symlog(x) = sign(x) * ln(|x| + 1)` a las entradas (particularmente recompensas) para manejar magnitudes diversas sin inestabilidad numérica.

### 2. Recurrent State-Space Model (RSSM)
El núcleo de la "memoria" del agente. Se descompone en:
- **Estado Determinista ($h_t$)**: Procesado por una GRU. Mantiene la historia a largo plazo.
- **Estado Estocástico ($z_t$)**: Representado por **variables categóricas discretas** (32 opciones x 32 clases).
  - *¿Por qué Discreto?* A diferencia de VAEs Gaussianos, las latentes discretas evitan que la varianza colapse a cero y obligan al modelo a capturar información semántica en lugar de ruido de píxeles.
- **Dinámica**:
  - *Posterior* $q(z_t | h_t, x_t)$: Infiere el estado actual viendo la imagen real.
  - *Prior* $p(z_t | h_t)$: Predice el estado actual basándose solo en la historia (imaginación).
  - **Objetivo KL**: Se minimiza la divergencia KL entre el Posterior y el Prior, enseñando al modelo a predecir el futuro sin ver la imagen.

### 3. Cabezas de Predicción (Heads)
- **Decoder**: Reconstruye la imagen desde el estado latente ($h_t, z_t$).
- **Predictor de Recompensa**: Estima $R_t$.
- **Predictor de Continuación**: Estima $\gamma_t$ (probabilidad de no morir/terminar).

## Aprendizaje de Conducta (Actor-Critic)

Una vez que el mundo se entiende, el agente aprende a actuar dentro de la simulación mental:
- **Imaginación**: Se despliegan trayectorias de ~15 pasos en el espacio latente usando el Prior.
- **Crítico (Value Network)**: Predice el retorno esperado (V-value) de los estados imaginados. Usa **Two-Hot Encoding** para predecir distribuciones de valor en lugar de una media simple, mejorando la estabilidad.
- **Actor (Policy Network)**: Maximiza el retorno esperado. Los gradientes fluyen a través de la dinámica del mundo (backpropagation through time), permitiendo una optimización mucho más eficiente que Policy Gradient estándar (REINFORCE).

## Innovaciones Clave de la V3
1.  **Geometric Balancing**: Equilibra automáticamente las pérdidas de reconstrucción, dinámica y KL sin necesidad de ajustar pesos ($\beta$) manualmente.
2.  **Free Bits**: Se reserva un presupuesto de entropía libre para evitar el sobreajuste del Prior.
3.  **Mastering from Pixels**: Eficiencia de muestra superior en entornos visuales complejos como Doom Deathmatch.

## ¿Por qué esta arquitectura para Doom?
- **Ocultación Parcial**: El RSSM recuerda enemigos que pasaron detrás de una pared (memoria recurrente).
- **Predicción de Riesgo**: Al simular el futuro, el agente "siente" el daño antes de recibirlo si entra en una zona peligrosa.
- **Generalización**: Al aprender la estructura del juego (física, reglas) en lugar de solo reacciones, se adapta mejor a mapas nuevos.
