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

# 2. Clone Dependencies
echo "Cloning/Updating nm512_dreamer repository..."
DREAMER_PATH="src/doom_agent/algorithms/dreamer_v3/nm512_dreamer"
rm -rf "$DREAMER_PATH"
git clone https://github.com/NM512/dreamerv3-torch "$DREAMER_PATH"
# Remove __init__.py if it exists to allow the project's flexible import structure
rm -f "$DREAMER_PATH/__init__.py"

# 3. Install Python Dependencies
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
export PYTHONPATH=$PROJECT_ROOT/src
echo "export PYTHONPATH=$PYTHONPATH" >> ~/.bashrc

# 4. Verify Installation & Setup Assets
echo "Verifying Installation and Setting up Assets..."
python3 -c "import torch; print(f'PyTorch available: {torch.cuda.is_available()} (Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"})')"
python3 -c "import vizdoom; print(f'VizDoom version: {vizdoom.__version__}')"

echo "Copying deathmatch.wad from vizdoom package..."
python3 -c "import vizdoom; import os; import shutil; src=os.path.join(os.path.dirname(vizdoom.__file__), 'scenarios', 'deathmatch.wad'); dst='src/doom_agent/scenarios/deathmatch.wad'; shutil.copy(src, dst) if os.path.exists(src) else print('Warning: deathmatch.wad not found in vizdoom package')"

echo "--- Setup Complete! ---"
echo ""
echo "To start training DreamerV3, run:"
echo "./run_deathmatch_5090.sh"
echo ""
echo "NOTE: On first run, VizDoom might compile assets. This is normal."
