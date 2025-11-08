#!/bin/bash
# Comprehensive WiLoR Hand Estimation Setup Script
# This script sets up the complete environment for 3D hand estimation using WiLoR
# Designed to run on Saturn Cloud nodes with NVIDIA GPUs

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================"
echo "WiLoR Hand Estimation Setup"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script requires Linux"
    exit 1
fi

print_status "Running on Linux"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_warning "nvidia-smi not found - GPU may not be available"
fi

echo ""
echo "Step 1: Installing Pixi package manager"
echo "--------------------------------------"

# Install pixi if not present
if ! command -v pixi &> /dev/null; then
    echo "Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash

    # Add pixi to PATH for current session
    export PATH="$HOME/.pixi/bin:$PATH"

    if command -v pixi &> /dev/null; then
        print_status "Pixi installed successfully"
    else
        print_error "Pixi installation failed"
        exit 1
    fi
else
    print_status "Pixi already installed ($(pixi --version))"
fi

echo ""
echo "Step 2: Setting up Pixi environment"
echo "------------------------------------"

# Install pixi environment
if [ ! -d ".pixi" ]; then
    echo "Creating pixi environment..."
    pixi install
    print_status "Pixi environment created"
else
    print_status "Pixi environment already exists"
    echo "Updating environment..."
    pixi install
fi

echo ""
echo "Step 3: Cloning WiLoR repository"
echo "---------------------------------"

# Clone WiLoR repository with submodules
if [ ! -d "WiLoR" ]; then
    echo "Cloning WiLoR repository from GitHub..."
    git clone --recursive https://github.com/rolpotamias/WiLoR.git
    print_status "WiLoR repository cloned"
else
    print_status "WiLoR repository already exists"
    echo "Updating submodules..."
    cd WiLoR
    git submodule update --init --recursive
    cd ..
fi

cd WiLoR

echo ""
echo "Step 4: Creating directory structure"
echo "-------------------------------------"

# Create necessary directories
mkdir -p mano_data pretrained_models
print_status "Directories created: mano_data, pretrained_models"

echo ""
echo "Step 5: Downloading MANO model files"
echo "-------------------------------------"

# Check for MANO models
MANO_DOWNLOADED=false

# Check common locations for MANO models
MANO_LOCATIONS=(
    "$HOME/Downloads/mano_v1_2/models"
    "$HOME/mano_v1_2/models"
    "$SCRIPT_DIR/mano_v1_2/models"
    "./mano_v1_2/models"
)

for location in "${MANO_LOCATIONS[@]}"; do
    if [ -f "$location/MANO_RIGHT.pkl" ]; then
        echo "Found MANO models in: $location"
        cp "$location/MANO_RIGHT.pkl" mano_data/
        print_status "MANO_RIGHT.pkl copied"

        if [ -f "$location/MANO_LEFT.pkl" ]; then
            cp "$location/MANO_LEFT.pkl" mano_data/
            print_status "MANO_LEFT.pkl copied"
        fi

        MANO_DOWNLOADED=true
        break
    fi
done

if [ "$MANO_DOWNLOADED" = false ]; then
    print_warning "MANO models not found locally"
    echo ""
    echo "MANO models must be downloaded manually from:"
    echo "https://mano.is.tue.mpg.de/"
    echo ""
    echo "After downloading mano_v1_2.zip:"
    echo "1. Extract the zip file"
    echo "2. Copy MANO_RIGHT.pkl and MANO_LEFT.pkl to: $SCRIPT_DIR/WiLoR/mano_data/"
    echo ""
    print_warning "Continuing setup - you'll need to add MANO models later"
fi

echo ""
echo "Step 6: Downloading pretrained models"
echo "--------------------------------------"

# Download detector model
if [ ! -f "pretrained_models/detector.pt" ]; then
    echo "Downloading hand detector model (52MB)..."
    wget --no-verbose --show-progress -P pretrained_models \
        https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
    print_status "Detector model downloaded"
else
    print_status "Detector model already exists"
fi

# Download WiLoR checkpoint
if [ ! -f "pretrained_models/wilor_final.ckpt" ]; then
    echo "Downloading WiLoR model checkpoint (2.4GB)..."
    wget --no-verbose --show-progress -P pretrained_models \
        https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt
    print_status "WiLoR model downloaded"
else
    print_status "WiLoR model already exists"
fi

cd ..

echo ""
echo "Step 7: Installing Python dependencies"
echo "---------------------------------------"

# Install Python dependencies from WiLoR requirements
if [ -f "WiLoR/requirements.txt" ]; then
    echo "Installing Python packages..."
    .pixi/envs/default/bin/python -m pip install -q -r WiLoR/requirements.txt
    print_status "Python dependencies installed"
else
    print_error "WiLoR/requirements.txt not found"
    exit 1
fi

echo ""
echo "Step 8: Setting up environment activation"
echo "------------------------------------------"

# Create/update activate.sh
cat > activate.sh << 'EOF'
#!/bin/bash
# WiLoR environment activation

# Add WiLoR to Python path
export PYTHONPATH="$PWD/WiLoR:$PYTHONPATH"

# Add pixi bin directory to PATH
export PATH="$PWD/.pixi/envs/default/bin:$PATH"

# Add PyTorch CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda

echo "WiLoR environment activated"
echo "PYTHONPATH: $PYTHONPATH"
EOF

chmod +x activate.sh
print_status "Environment activation script created"

echo ""
echo "Step 9: Verifying installation"
echo "-------------------------------"

# Check Python version
PYTHON_VERSION=$(.pixi/envs/default/bin/python --version 2>&1)
print_status "Python: $PYTHON_VERSION"

# Check PyTorch
TORCH_VERSION=$(.pixi/envs/default/bin/python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>&1)
if [ $? -eq 0 ]; then
    print_status "$TORCH_VERSION"
else
    print_warning "PyTorch check failed"
fi

# Check CUDA availability
CUDA_AVAILABLE=$(.pixi/envs/default/bin/python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CUDA not available')" 2>&1)
if [ $? -eq 0 ]; then
    if [[ "$CUDA_AVAILABLE" == *"CUDA available"* ]]; then
        print_status "$CUDA_AVAILABLE"
        GPU_COUNT=$(.pixi/envs/default/bin/python -c "import torch; print(torch.cuda.device_count())" 2>&1)
        print_status "GPU count: $GPU_COUNT"
    else
        print_warning "$CUDA_AVAILABLE"
    fi
fi

# Check if models exist
echo ""
echo "Model files:"
if [ -f "WiLoR/mano_data/MANO_RIGHT.pkl" ]; then
    print_status "MANO_RIGHT.pkl ($(du -h WiLoR/mano_data/MANO_RIGHT.pkl | cut -f1))"
else
    print_warning "MANO_RIGHT.pkl missing"
fi

if [ -f "WiLoR/mano_data/MANO_LEFT.pkl" ]; then
    print_status "MANO_LEFT.pkl ($(du -h WiLoR/mano_data/MANO_LEFT.pkl | cut -f1))"
else
    print_warning "MANO_LEFT.pkl missing"
fi

if [ -f "WiLoR/pretrained_models/detector.pt" ]; then
    print_status "detector.pt ($(du -h WiLoR/pretrained_models/detector.pt | cut -f1))"
else
    print_warning "detector.pt missing"
fi

if [ -f "WiLoR/pretrained_models/wilor_final.ckpt" ]; then
    print_status "wilor_final.ckpt ($(du -h WiLoR/pretrained_models/wilor_final.ckpt | cut -f1))"
else
    print_warning "wilor_final.ckpt missing"
fi

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Environment is ready to use."
echo ""
echo "To activate the environment:"
echo "  source activate.sh"
echo ""
echo "To process video frames:"
echo "  .pixi/envs/default/bin/python process_video_frames.py -i video.mp4 -o output_dir --all-frames"
echo ""
echo "To extract 3D hand data:"
echo "  .pixi/envs/default/bin/python extract_hand_3d.py -i video.mp4 -o output_dir --output-video trajectory.mp4"
echo ""
echo "For more usage examples, see README.md"
echo ""

if [ "$MANO_DOWNLOADED" = false ]; then
    echo "========================================"
    echo "IMPORTANT: MANO Models Required"
    echo "========================================"
    echo ""
    print_warning "You still need to download MANO models manually"
    echo ""
    echo "1. Visit: https://mano.is.tue.mpg.de/"
    echo "2. Register and download mano_v1_2.zip"
    echo "3. Extract and copy MANO_RIGHT.pkl and MANO_LEFT.pkl to:"
    echo "   $SCRIPT_DIR/WiLoR/mano_data/"
    echo ""
fi
