#!/bin/bash
# Video Depth Anything setup script

set -e

echo "Setting up Video Depth Anything..."

# Clone repository
if [ ! -d "Video-Depth-Anything" ]; then
    echo "Cloning Video-Depth-Anything repository..."
    git clone https://github.com/DepthAnything/Video-Depth-Anything
else
    echo "✓ Video-Depth-Anything repository already exists"
fi

# Note: Python dependencies are managed by pixi.toml
# The following packages are automatically installed:
# - opencv-python (via conda opencv package)
# - imageio, imageio-ffmpeg, einops, tqdm (via conda)
# - decord, easydict (via pip through pypi-dependencies)
# - xformers is intentionally NOT installed (force-disabled for compatibility)
echo "✓ Python dependencies managed by pixi (see pixi.toml)"

# Download model weights
echo "Downloading model weights..."
cd Video-Depth-Anything

if [ ! -f "get_weights.sh" ]; then
    echo "Creating weight download script..."
    cat > get_weights.sh << 'EOF'
#!/bin/bash
mkdir -p checkpoints
cd checkpoints

# Download Depth Anything V2 models
echo "Downloading Depth Anything V2 models..."

# Small model (28M parameters)
if [ ! -f "depth_anything_v2_vits.pth" ]; then
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
fi

# Base model (113M parameters)
if [ ! -f "depth_anything_v2_vitb.pth" ]; then
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
fi

# Large model (381M parameters)
if [ ! -f "depth_anything_v2_vitl.pth" ]; then
    wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
fi

echo "✓ Model weights downloaded"
EOF
    chmod +x get_weights.sh
fi

bash get_weights.sh

cd ..

echo ""
echo "✓ Setup complete!"
echo ""
echo "Model checkpoints are in: Video-Depth-Anything/checkpoints/"
echo "  - depth_anything_v2_vits.pth (Small, 28M params)"
echo "  - depth_anything_v2_vitb.pth (Base, 113M params)"
echo "  - depth_anything_v2_vitl.pth (Large, 381M params)"
