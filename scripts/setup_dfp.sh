#!/bin/bash
# scripts/setup_dfp.sh - Automated installer and model fetcher for Intel DFP benchmark

set -e

REPO_DIR="external/dfp"
BRANCH="pretrained_models"
MODEL_PATH="examples/D3_battle/checkpoints/2017_04_08_10_44_20"

echo "🚀 Setting up Intel DFP (Direct Future Prediction) Benchmark..."

# 1. Clone/Fetch repository
if [ ! -d "$REPO_DIR" ]; then
    echo "📦 Cloning DFP repository..."
    mkdir -p external
    git clone --depth 1 https://github.com/isl-org/DirectFuturePrediction.git "$REPO_DIR"
else
    echo "✅ DFP repository folder exists."
fi

cd "$REPO_DIR"

# 2. Fetch pre-trained models branch
echo "📥 Fetching pre-trained models branch..."
git fetch origin "$BRANCH:$BRANCH" --quiet || git fetch origin "$BRANCH" --quiet

# 3. Extract specific model files
echo "🗂️ Extracting D3_battle model files..."
git checkout "$BRANCH" -- "$MODEL_PATH"

# Move them to a cleaner location
mkdir -p pretrained
cp -r "$MODEL_PATH/"* pretrained/
rm -rf examples/D3_battle/checkpoints

# 4. Apply TensorFlow 2 Compatibility Patches
echo "🛠️ Applying compatibility patches for TensorFlow 2..."

FILES=("./DFP"/*.py)
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf/g' "$f"
    fi
done

# Fix dtype error in exponential_decay by replacing it with a constant for inference
sed -i 's/self.tf_learning_rate = tf.train.exponential_decay.*/self.tf_learning_rate = tf.constant(self.init_learning_rate, dtype=tf.float32)/g' "./DFP/agent.py"

# Fix broadcasting error in act_net
sed -i 's/curr_objective_coeffs\[:,None,:\]/curr_objective_coeffs\[:,None,self.objective_indices\]/g' "./DFP/future_predictor_agent_advantage.py"

# Disable V2 behavior in the main DFP package initialization
INIT_FILE="./DFP/__init__.py"
echo "import tensorflow.compat.v1 as tf; tf.disable_v2_behavior()" > "$INIT_FILE"

echo "🎯 DFP setup complete!"
echo "   Models are located in $REPO_DIR/pretrained/"
