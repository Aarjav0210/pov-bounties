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
