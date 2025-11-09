#!/bin/bash
# Grounded Segment Anything environment activation

# Add PyTorch CUDA libraries to LD_LIBRARY_PATH for compiled extensions
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda

echo "Grounded-SAM environment activated"
echo "Using pre-built GroundingDINO package"
