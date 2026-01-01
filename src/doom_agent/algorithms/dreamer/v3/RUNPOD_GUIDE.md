# Configuración para Entrenamiento en RunPod (GPU Cloud)

Este proyecto está listo para ejecutarse en instancias de RunPod (RTX 3090, 4090 o 6000 Ada).

## 1. Setup Recomendado (RunPod Pod)
*   **Imagen**: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` (o superior).
*   **VRAM**: Mínimo 24GB (3090/4090). Recomendado 48GB (6000 Ada) para el perfil Peak.
*   **Espacio en Disco**: Mínimo 50GB para el Replay Buffer y checkpoints.

## 2. Instalación en RunPod
Ejecuta estos comandos al iniciar tu Pod:

```bash
# Instalar dependencias del sistema para VizDoom
apt-get update
apt-get install -y build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
    nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev \
    libgme-dev libopenal-dev timidity libwildmidi-dev libopenjp2-7-dev

# Instalar dependencias de Python
pip install vizdoom opencv-python hydra-core omegaconf wandb ruamel.yaml tensorboard

# Clonar repositorio y configurar path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

## 3. Entrenamiento con Alto Rendimiento
Para una **RTX 6000 Ada** (Recomendado):
```bash
python src/doom_agent/algorithms/dreamer/v3/train.py \
    hardware=rtx6000ada \
    wandb.enabled=true \
    wandb.project=doom-dreamer
```

Para una **RTX 3090 / 4090**:
```bash
python src/doom_agent/algorithms/dreamer/v3/train.py \
    hardware=rtx3060 \
    agent.batch_size=64 \
    agent.n_envs=8
```

## 4. Notas de RunPod
- **Headless**: El script ya incluye `os.environ['SDL_AUDIODRIVER'] = 'dummy'` para funcionar sin pantalla ni sonido.
- **WandB**: Recomendamos usar Weights & Biases para monitorizar el progreso remotamente, ya que RunPod suele cerrar puertos de TensorBoard.
- **Auto-Save**: Los checkpoints se guardan automáticamente cada etapa en `outputs/`.
