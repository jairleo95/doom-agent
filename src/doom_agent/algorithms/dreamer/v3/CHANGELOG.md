# DreamerV3 Technical Changelog

Este documento detalla todas las mejoras y correcciones realizadas en la implementación de DreamerV3 para VizDoom.

## 1. Correcciones de Arquitectura y Carga

### [Fix] Mapeo de `num_actions`
*   **Cambio**: Se añadió explícitamente `num_actions` a la configuración del agente en `train.py` y `visualize.py`.
*   **Por qué**: La biblioteca interna `nm512_dreamer` utiliza la clave `num_actions` para definir la dimensión de salida del actor y el modelo de mundo. Sin esto, el script fallaba con un `KeyError`.
*   **Código**:
    ```python
    agent_config = {
        'action_dim': len(actions),
        'num_actions': len(actions), # Requerido por el WorldModel interno
        # ...
    }
    ```

### [Fix] Carga Robusta de Checkpoints (Namespaces de `torch.compile`)
*   **Cambio**: Implementación de mapeo recursivo de llaves en `agent.py:load()`.
*   **Por qué**: Al entrenar con `torch.compile`, PyTorch añade el prefijo `_orig_mod.` a todas las capas. Si intentamos cargar ese archivo en un agente no compilado (o viceversa), ocurre un error de "Missing/Unexpected keys". El nuevo cargador limpia los nombres y los empareja dinámicamente.
*   **Código**:
    ```python
    model_keys = self.agent.state_dict().keys()
    clean_to_model = {k.replace("_orig_mod.", ""): k for k in model_keys}
    
    final_state_dict = {}
    for k, v in checkpoint_state_dict.items():
        clean_name = k.replace("_orig_mod.", "")
        if clean_name in clean_to_model:
            final_state_dict[clean_to_model[clean_name]] = v
    ```

---

## 2. Optimización del Comportamiento (Reward Shaping)

### [Improvement] Penalización por Desperdicio de Munición
*   **Cambio**: Aumento drástico de `ammo_penalty` (de 0.03 a 0.20 en Expert).
*   **Por qué**: El agente se quedaba "disparando a la pared". Al disparar sin criterio, encontraba recompensas de forma aleatoria. Con una penalización alta, el agente aprende que cada bala fallada es un error costoso, forzándolo a buscar enemigos antes de apretar el gatillo.

### [Improvement] Penalización por Daño Recibido
*   **Cambio**: Aumento de `health_penalty` a 0.50 en etapas avanzadas.
*   **Por qué**: Incentiva al agente a usar movimientos laterales (strafing) y cobertura. Si el coste de ser golpeado es alto, la supervivencia se convierte en la prioridad #1.

---

## 3. Estrategia de Exploración Activa

### [Feature] Integración de `Plan2Explore`
*   **Cambio**: Cambio de `expl_behavior` de 'greedy' a 'plan2explore' y configuración de `expl_until: 500k`.
*   **Por qué**: El agente tendía a volverse "conservador" muy rápido. `Plan2Explore` genera una curiosidad intrínseca basada en el desacuerdo del modelo (disagreement). El agente busca activamente situaciones que su modelo de mundo aún no sabe predecir bien.
*   **Código**:
    ```python
    'expl_behavior': 'plan2explore',
    'expl_until': 500_000, # Fase de descubrimiento puro
    ```

---

## 4. Rendimiento y Hardware

### [Optimization] Incremento del Training Ratio
*   **Cambio**: Ajuste de `train_ratio` a 1024 (o `train_every: 64`).
*   **Por qué**: Aprovecha el gran VRAM de GPUs como la RTX 6000 Ada. Al entrenar con más frecuencia y mayor intensidad, el agente procesa más "imaginación" por cada paso real, acelerando la convergencia.

### [Optimization] Estabilización de ETA (EMA)
*   **Cambio**: Implementación de Media Móvil Exponencial (EMA) para el cálculo de FPS.
*   **Por qué**: El FPS de Dreamer fluctúa mucho entre las fases de recolección y entrenamiento. El EMA suaviza estas variaciones para dar una estimación de tiempo de finalización realista y estable.
*   **Código**:
    ```python
    alpha = 0.3
    self.ema_fps = alpha * current_fps + (1 - alpha) * self.ema_fps
    ```

---

## 5. Herramientas de Visualización

### [New] Script `visualize.py`
*   **Cambio**: Creación de un entorno de inferencia en tiempo real optimizado.
*   **Por qué**: Permite depurar el comportamiento del agente "en vivo", detectar errores tácticos (como disparar a paredes) y verificar que el cargador de checkpoints funciona correctamente.
*   **Feature**: Auto-detección del último checkpoint experto disponible.

---

## 6. Mejoras Visuales y de Datos (High-Fidelity)

### [Fix] Estabilización de Video (PIL.Image Crash)
*   **Cambio**: Implementación de `robust_transpose` en `doom_envs.py`.
*   **Por qué**: El sistema a veces intentaba procesar imágenes con dimensiones intercambiadas (320x3x240 en lugar de 320x240x3), lo que causaba un error fatal en Pillow. El nuevo helper asegura que el formato siempre sea `(H, W, C)` antes de guardarlo.

### [Feature] Aumento por Simetría (Mirror Learning)
*   **Cambio**: Integración de `horizontal_flip` en el `ReplayBuffer` y re-mapeo de acciones en `train.py`.
*   **Por qué**: Duplica la eficiencia de los datos al permitir que el agente aprenda simultáneamente de situaciones "reales" y sus versiones espejo. Se ajustan automáticamente los giros (izquierda <-> derecha) para que la lógica de combate sea consistente.

### [Feature] Videos de "Sueños" (Imagination Logging)
*   **Cambio**: Nueva callback `ImaginationVideoCallback`.
*   **Por qué**: Permite visualizar la arquitectura de "sueño" de DreamerV3 en TensorBoard. Muestra la secuencia real, la predicción del modelo de mundo y el mapa de error, permitiendo diagnosticar si el agente está "alucinando" o si entiende bien su entorno.

### [Feature] Analítica de Gameplay Detallada
*   **Cambio**: Registro de `frags`, `health` y `ammo` por episodio.
*   **Por qué**: En lugar de ver solo una curva de recompensa genérica, ahora podemos ver curvas de precisión (frags), supervivencia (salud restante) y eficiencia (munición consumida), facilitando el tuneo de recompensas.

---

## 7. Refactorización del Directorio de Algoritmos

### [Refactor] Reorganización Jerárquica
*   **Cambio**: Agrupación de todas las versiones de algoritmos en familias (`ppo/`, `a2c/`, `dqn/`, `dreamer/`).
*   **Por qué**: Mejora la navegabilidad del proyecto. DreamerV3 ahora reside en `src/doom_agent/algorithms/dreamer/v3/`.
---

## 8. Arquitectura Moderna y DevOps (SOLID & Hydra)

### [Refactor] Refactorización SOLID (DreamerV3Trainer)
*   **Cambio**: Migración de la lógica de entrenamiento monolítica en `train.py` a clases especializadas en `trainer.py` y `experiment.py`.
*   **Por qué**: Mejora la testabilidad y el mantenimiento. `DreamerV3Trainer` se encarga exclusivamente de la orquestación del agente y los entornos, mientras que `ExperimentManager` gestiona el sistema de archivos y metadatos.
*   **Código**:
    ```python
    trainer = DreamerV3Trainer(cfg, exp, curriculum, actions)
    trainer.run()
    ```

### [Feature] Integración de Hydra Config
*   **Cambio**: Implementación de Hydra para el manejo de configuraciones jerárquicas.
*   **Por qué**: Permite desacoplar los parámetros del hardware (RTX 3060 vs 6000 Ada) de la lógica del escenario. Facilita las pruebas locales y en la nube mediante overrides de línea de comandos.
*   **Uso**: `python train.py scenario=deathmatch hardware=rtx3060`

### [Feature] W&B Artifacts y Checkpointing en la Nube
*   **Cambio**: Automatización de la subida de modelos a Weights & Biases.
*   **Por qué**: Asegura que el progreso del entrenamiento esté respaldado fuera del servidor local. Se suben automáticamente el "Mejor Modelo" basado en evaluación y el modelo final de cada etapa.
*   **Código**: `wandb.log_artifact(path, name=name, type='model')`

### [Improvement] Suite de Pruebas Unificada (PyTest)
*   **Cambio**: Sincronización completa de los tests con la nueva arquitectura y entrenamiento RGB.
*   **Por qué**: Garantiza que los cambios en la arquitectura no introduzcan regresiones. Se añadieron pruebas de orquestación para validar el flujo `ExperimentManager -> Trainer`.
