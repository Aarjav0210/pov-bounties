#!/bin/bash
# Video Depth Anything environment activation

# Add repository to Python path
export PYTHONPATH="$PWD/Video-Depth-Anything:$PYTHONPATH"

# Add PyTorch CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda

echo "Video-Depth-Anything environment activated"
echo "PYTHONPATH: $PYTHONPATH"
