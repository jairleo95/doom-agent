#!/bin/bash
# scripts/setup_arnold.sh - Automated installer and patcher for Arnold benchmark

set -e

REPO_DIR="external/arnold"
MODEL_URL="https://github.com/glample/Arnold/raw/master/pretrained"

echo "🚀 Setting up Arnold SOTA Benchmark..."

# 1. Clone repository if not exists
if [ ! -d "$REPO_DIR" ]; then
    echo "📦 Cloning Arnold repository..."
    mkdir -p external
    git clone https://github.com/glample/Arnold.git "$REPO_DIR"
else
    echo "✅ Arnold repository already exists."
fi

# 2. Download pre-trained models
echo "📥 Downloading pre-trained models..."
mkdir -p "$REPO_DIR/pretrained"
MODELS=("vizdoom_2017_track2.pth" "vizdoom_2017_track1.pth" "deathmatch_shotgun.pth")

for model in "${MODELS[@]}"; do
    if [ ! -f "$REPO_DIR/pretrained/$model" ]; then
        echo "   Downloading $model..."
        wget "$MODEL_URL/$model" -O "$REPO_DIR/pretrained/$model" --quiet
    fi
done

# 3. Apply PyTorch 2.5+ Compatibility Patches
echo "🛠️ Applying compatibility patches..."

# Patch: BucketedEmbedding division (Float -> Long)
SED_EMB="s/indices.div(self.bucket_size)/indices \/\/ self.bucket_size/g"
sed -i "$SED_EMB" "$REPO_DIR/src/model/bucketed_embedding.py"

# Patch: Tensor indexing (Scalar indexing fix)
SED_IDX="s/scores.data.max(0)\[1\]\[0\]/scores.data.max(0)\[1\].item()/g"
sed -i "$SED_IDX" "$REPO_DIR/src/model/dqn/base.py"

echo "🎯 Arnold setup complete! You can now run the benchmark with:"
echo "   python scripts/test_reward_shaping.py --arnold"
