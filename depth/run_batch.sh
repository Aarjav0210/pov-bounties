#!/bin/bash
# Batch process all videos in inputs directory
# Usage: ./run_batch.sh [encoder]
#   encoder: vits (fastest), vitb (balanced), or vitl (best quality, default)

set -e

# Determine the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the pixi environment
echo "Activating environment..."
source "${SCRIPT_DIR}/activate.sh"

# Get encoder from argument or use default
ENCODER="${1:-vitl}"

# Valid encoders
if [[ ! "$ENCODER" =~ ^(vits|vitb|vitl)$ ]]; then
    echo "Error: Invalid encoder '$ENCODER'"
    echo "Usage: $0 [vits|vitb|vitl]"
    echo "  vits - Fast, less VRAM (6.8GB)"
    echo "  vitb - Balanced"
    echo "  vitl - Best quality (23.6GB VRAM) [default]"
    exit 1
fi

echo "Using encoder: $ENCODER"

# Update the encoder in the Python script temporarily or pass as environment variable
export DEPTH_ENCODER="$ENCODER"

# Run the batch processing script
python3 "${SCRIPT_DIR}/batch_process_videos.py"
