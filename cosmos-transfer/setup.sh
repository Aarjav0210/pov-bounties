#!/bin/bash
set -e

echo "Checking GPU..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Clone repo
if [ ! -d "cosmos-transfer2.5" ]; then
    git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
fi

cd cosmos-transfer2.5

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python 3.10 virtual environment..."
    python3.10 -m venv venv
fi

# Activate and install
echo "Installing cosmos-transfer2.5 with CUDA extras..."
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install uv
python -m uv pip install -e ".[cu128_torch271]"
python -m uv pip install "huggingface_hub[cli]"
python -m uv pip install timm opencv-python

echo "Setup complete! Please authenticate with HuggingFace:"
echo "  huggingface-cli login"
echo "  Visit: https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B"

cd ..
