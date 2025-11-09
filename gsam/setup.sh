#!/bin/bash
# Grounded Segment Anything setup script
set -e

echo "========================================"
echo "Grounded Segment Anything Setup"
echo "========================================"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found"
else
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Get absolute path to pixi python
PYTHON_BIN="$(pwd)/.pixi/envs/default/bin/python"

echo ""
echo "Cloning repositories..."

# Clone Segment Anything
if [ ! -d "segment-anything" ]; then
    echo "Cloning Segment Anything..."
    git clone https://github.com/facebookresearch/segment-anything.git
else
    echo "segment-anything already cloned"
fi

# Clone Grounding DINO
if [ ! -d "GroundingDINO" ]; then
    echo "Cloning Grounding DINO..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
else
    echo "GroundingDINO already cloned"
fi

# Clone Grounded-Segment-Anything for reference
if [ ! -d "Grounded-Segment-Anything" ]; then
    echo "Cloning Grounded-Segment-Anything..."
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
else
    echo "Grounded-Segment-Anything already cloned"
fi

echo ""
echo "Installing dependencies..."

# Install Segment Anything
echo "Installing Segment Anything..."
cd segment-anything
"$PYTHON_BIN" -m pip install -e .
cd ..

# Install GroundingDINO
echo "Installing GroundingDINO..."
cd GroundingDINO
# Set CUDA environment
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda
"$PYTHON_BIN" -m pip install --no-build-isolation -e .
cd ..

# Install additional dependencies
echo "Installing additional packages..."
"$PYTHON_BIN" -m pip install opencv-python pycocotools matplotlib supervision

# Download model checkpoints
echo ""
echo "Downloading model checkpoints..."
mkdir -p weights

# Download SAM checkpoint (ViT-H)
if [ ! -f "weights/sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM ViT-H checkpoint..."
    wget -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo "SAM checkpoint already downloaded"
fi

# Download Grounding DINO checkpoint
if [ ! -f "weights/groundingdino_swint_ogc.pth" ]; then
    echo "Downloading Grounding DINO checkpoint..."
    wget -P weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "Grounding DINO checkpoint already downloaded"
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Model checkpoints saved to weights/"
echo "Run segmentation with:"
echo "  pixi run segment -i video.mp4 -p 'hand . object . table'"
echo ""
