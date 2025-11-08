#!/bin/bash

# Installation script for Video Validation API
# This handles the correct installation order for dependencies

set -e

echo "🔧 Installing Video Validation API Dependencies"
echo "================================================"
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   It's recommended to use a virtual environment"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Create a venv with:"
        echo "  python -m venv pov-env"
        echo "  source pov-env/bin/activate"
        exit 1
    fi
fi

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip
echo ""

# Step 1: Install PyTorch and torchvision first (required for flash-attn and Qwen2VL)
echo "🔥 Step 1/3: Installing PyTorch and torchvision..."
pip install torch>=2.0.0 torchvision>=0.15.0
echo "✅ PyTorch and torchvision installed"
echo ""

# Step 2: Install other core dependencies (without flash-attn)
echo "📚 Step 2/3: Installing core dependencies..."
pip install \
    transformers>=4.37.0 \
    opencv-python>=4.8.0 \
    numpy>=1.24.0 \
    Pillow>=10.0.0 \
    ruptures>=1.1.0 \
    tqdm>=4.65.0 \
    accelerate>=0.25.0 \
    av>=10.0.0 \
    qwen-vl-utils>=0.0.1 \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    python-multipart>=0.0.6
echo "✅ Core dependencies installed"
echo ""

# Step 3: Install flash-attn (requires torch and CUDA)
echo "⚡ Step 3/3: Installing flash-attn (this may take a few minutes)..."
echo "   Note: This requires NVCC (CUDA compiler) to be available"
echo ""

if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "   Found NVCC version: $NVCC_VERSION"
    echo ""
    
    # Try to install flash-attn
    if pip install flash-attn>=2.3.0 --no-build-isolation; then
        echo "✅ flash-attn installed successfully"
    else
        echo "⚠️  flash-attn installation failed"
        echo "   This is optional but improves performance"
        echo "   The API will still work without it"
        echo ""
        echo "   To install manually later:"
        echo "   pip install flash-attn --no-build-isolation"
    fi
else
    echo "⚠️  NVCC not found - skipping flash-attn installation"
    echo "   flash-attn is optional but improves performance"
    echo "   The API will still work without it"
    echo ""
    echo "   If you have CUDA installed, make sure NVCC is in your PATH:"
    echo "   export PATH=/usr/local/cuda/bin:\$PATH"
fi

echo ""
echo "================================================"
echo "✅ Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Make sure config.py is configured (or use config.example.py as template)"
echo "  2. Start the API: bash start_api.sh"
echo "  3. Test the API: ./test_api.sh /path/to/video.mp4 \"task description\""
echo ""

