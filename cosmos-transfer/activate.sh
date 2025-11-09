#!/bin/bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -d "$PWD/cosmos-transfer2.5/venv" ]; then
    source "$PWD/cosmos-transfer2.5/venv/bin/activate"
    export PYTHONPATH="$PWD/cosmos-transfer2.5:$PYTHONPATH"
    echo "Cosmos Transfer 2.5 environment activated"
    echo "Python: $(which python)"
    echo "HF_HOME: $HF_HOME"
else
    echo "ERROR: Virtual environment not found. Run 'bash setup.sh' first."
fi
