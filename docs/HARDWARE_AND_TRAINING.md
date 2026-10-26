# ⚡ Guía de Entrenamiento y Perfiles de Hardware

Este documento describe la configuración de infraestructura, perfiles de aceleración por GPU, gestión de memoria y el sistema de seguimiento de experimentos para [doom-agent](file:///home/darkstar/Workspace/ai/rl/doom-agent/README.md).

---

## 1. Sistema de Configuración con Hydra

Los entrenamientos de DreamerV3 se parametrizan mediante [Hydra](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/conf/config.yaml), lo que permite sobreescribir cualquier variable desde la línea de comandos sin alterar el código fuente:

```bash
python src/doom_agent/algorithms/dreamer/v3/train.py \
  scenario=deathmatch \
  hardware=rtx3060 \
  wandb.enabled=true \
  device=cuda
```

### Jerarquía de Configuración (`conf/`)

- [conf/config.yaml](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/conf/config.yaml): Parámetros raíz (pasos de entrenamiento, semillas, directorios de log).
- [conf/scenario/](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/conf/scenario): Definición de escenarios (*deathmatch*, *deadly_corridor*, *defend_the_center*).
- [conf/hardware/](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/conf/hardware): Especificaciones de batch size, paralelismo y gestión de VRAM.

---

## 2. Perfiles de Hardware Soportados

| Perfil | VRAM | Caso de Uso | Script Preparado | Configuración Clave |
|---|---|---|---|---|
| **rtx3060** | 12 GB | Desarrollo local y depuración rápida | [test_local_3060.sh](file:///home/darkstar/Workspace/ai/rl/doom-agent/test_local_3060.sh) | `expandable_segments:True`, batch conservador para evitar OOM |
| **rtx5090** | 32 GB | Estación de trabajo de alto rendimiento | [run_deathmatch_5090.sh](file:///home/darkstar/Workspace/ai/rl/doom-agent/run_deathmatch_5090.sh) | Batch size ampliado y mayor número de entornos concurrentes |
| **rtx6000ada** | 48 GB | Workstation profesional para runs prolongados | [run_deathmatch_6000ada.sh](file:///home/darkstar/Workspace/ai/rl/doom-agent/run_deathmatch_6000ada.sh) | Mayor horizonte de imaginación en el RSSM |
| **h200** | 141 GB HBM3e | Entrenamiento masivo en clúster cloud | [train_h200_sota.sh](file:///home/darkstar/Workspace/ai/rl/doom-agent/train_h200_sota.sh) | Capacidad para millones de transiciones en Replay Buffer rápido |

---

## 3. Despliegue en Servidores Cloud (RunPod / Vast.ai / AWS)

En entornos de computación sin servidor gráfico o contenedores remotos, ViZDoom requiere dependencias de sistema y controladores de audio simulados para evitar cuelgues.

### Preparación del Entorno Cloud

El script [setup_cloud.sh](file:///home/darkstar/Workspace/ai/rl/doom-agent/setup_cloud.sh) automatiza:
1. Instalación de paquetes de sistema (`libboost`, `libgl1`, `ffmpeg`).
2. Configuración de controladores de audio nulos para entornos *headless*:
   ```bash
   export SDL_AUDIODRIVER=dummy
   export ALSOFT_DRIVERS=null
   ```
3. Creación del entorno e instalación de dependencias desde [requirements_cloud.txt](file:///home/darkstar/Workspace/ai/rl/doom-agent/requirements_cloud.txt).

```bash
# Ejecutar configuración automática en una instancia limpia
bash setup_cloud.sh
```

---

## 4. Monitoreo, Telemetría y Visualización

### 4.1. Weights & Biases (WandB)

El entrenamiento registra en tiempo real métricas de rendimiento y dinámicas del agente:
- **Rendimiento**: FPS de simulación, FPS de entrenamiento de red, uso de memoria GPU.
- **Gameplay**: Recompensa media acumulada, frags conseguidos, ratio K/D, duración del episodio.
- **World Model**: Pérdida de reconstrucción de imágenes, error de predicción de dinámica latente y entropía de la política.

Para habilitarlo:
```bash
python src/doom_agent/algorithms/dreamer/v3/train.py wandb.enabled=true wandb.project="doom-dreamer"
```

### 4.2. Grabación de Vídeos (Episodios Reales vs. Imaginados)

El agente cuenta con callbacks especializados en [callbacks](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/callbacks):
- **[VideoRecorderCallback](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/callbacks/video.py)**: Guarda clips MP4/GIF del agente jugando en el simulador.
- **[ImaginationVideoCallback](file:///home/darkstar/Workspace/ai/rl/doom-agent/src/doom_agent/algorithms/dreamer/v3/callbacks/imagination.py)**: Reconstruye mediante el decodificador de imágenes del RSSM las trayectorias ficticias que el agente "imagina" durante la optimización Actor-Crítico.

### 4.3. Herramientas de Inspección Local

- **Visualizar checkpoints locales**:
  ```bash
  python src/doom_agent/algorithms/dreamer/v3/visualize.py --checkpoint results/latest_model.ckpt
  ```
- **Monitorear en directo el agente en ejecución**:
  ```bash
  bash watch_dreamer.sh
  ```
- **Descargar y evaluar artefactos de WandB**:
  ```bash
  bash visualize_from_wandb.sh
  ```
