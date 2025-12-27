#!/bin/bash

# Setup script for DreamerV3 and PPO v5 training on GPU Cloud (RunPod, Lambda, etc.)
# This script assumes an Ubuntu-based image with NVIDIA drivers and CUDA pre-installed.

set -e

echo "--- Starting Automatic Cloud Setup ---"

# 1. Install System Dependencies for VizDoom
echo "Installing System Dependencies (apt)..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    zlib1g-dev \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    cmake \
    git \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    timidity \
    libwildmidi-dev \
    libboost-all-dev \
    python3-dev \
    python3-pip \
    wget \
    unzip \
    libxext-dev \
    libxrender-dev \
    libgl1-mesa-dev

# 2. Install Python Dependencies
echo "Installing Python Packages (pip)..."
pip install --upgrade pip
if [ -f "requirements_cloud.txt" ]; then
    pip install -r requirements_cloud.txt
else
    echo "Error: requirements_cloud.txt not found!"
    exit 1
fi

# 3. Setup Project Paths
echo "Exporting PYTHONPATH..."
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT/src/doom_agent/algorithms/dreamer_v3/nm512_dreamer
echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bashrc

# 4. Verify Installation
echo "Verifying Installation..."
python3 -c "import torch; print(f'PyTorch available: {torch.cuda.is_available()} (Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"})')"
python3 -c "import vizdoom; print(f'VizDoom version: {vizdoom.__version__}')"

echo "--- Setup Complete! ---"
echo ""
echo "To start training DreamerV3, run:"
echo "export PYTHONPATH=$(pwd)/src:$(pwd)/src/doom_agent/algorithms/dreamer_v3/nm512_dreamer"
echo "python src/doom_agent/algorithms/dreamer_v3/train.py --scenario deadly_corridor --n-envs 16 --device cuda"
echo ""
echo "NOTE: On first run, VizDoom might compile assets. This is normal."
