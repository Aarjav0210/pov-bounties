#!/bin/bash
# Grounded Segment Anything setup script (using pre-built packages)
set -e

echo "========================================"
echo "Grounded Segment Anything Setup"
echo "Using pre-built GroundingDINO packages"
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
echo "PyTorch version:"
"$PYTHON_BIN" -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

echo ""
echo "Installing Segment Anything..."
"$PYTHON_BIN" -m pip install git+https://github.com/facebookresearch/segment-anything.git

echo ""
echo "Installing GroundingDINO (pre-built)..."
# Try multiple pre-built options in order of preference

# Option 1: Roboflow's pre-built version (most recent)
echo "Trying rf-groundingdino..."
if "$PYTHON_BIN" -m pip install rf-groundingdino --no-deps 2>/dev/null; then
    echo "✓ Successfully installed rf-groundingdino"
    GROUNDING_DINO_INSTALLED=true
else
    echo "✗ rf-groundingdino installation failed"
    GROUNDING_DINO_INSTALLED=false
fi

# Option 2: Try autodistill version if first failed
if [ "$GROUNDING_DINO_INSTALLED" = false ]; then
    echo "Trying autodistill-grounding-dino..."
    if "$PYTHON_BIN" -m pip install autodistill-grounding-dino 2>/dev/null; then
        echo "✓ Successfully installed autodistill-grounding-dino"
        GROUNDING_DINO_INSTALLED=true
    else
        echo "✗ autodistill-grounding-dino installation failed"
    fi
fi

# Option 3: Clone and install dependencies only (no compilation)
if [ "$GROUNDING_DINO_INSTALLED" = false ]; then
    echo "Trying to clone GroundingDINO and install dependencies..."
    if [ ! -d "GroundingDINO" ]; then
        git clone https://github.com/IDEA-Research/GroundingDINO.git
    fi
    cd GroundingDINO
    # Install dependencies without building C++ extensions
    "$PYTHON_BIN" -m pip install --no-deps transformers addict yapf timm supervision
    cd ..
    echo "⚠ GroundingDINO installed in dependency-only mode (may have limited functionality)"
    GROUNDING_DINO_INSTALLED=partial
fi

# Install required dependencies
echo ""
echo "Installing additional packages..."
"$PYTHON_BIN" -m pip install opencv-python pycocotools matplotlib

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
echo "GroundingDINO status: $GROUNDING_DINO_INSTALLED"
echo "Model checkpoints saved to weights/"
echo "Run segmentation with:"
echo "  pixi run segment -i video.mp4 -p 'hand . object . table'"
echo ""
