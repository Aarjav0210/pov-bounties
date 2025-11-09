#!/usr/bin/env python3
"""NVIDIA Cosmos Transfer 2.5 - Video Domain Randomization"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path


def create_transfer_config(
    video_path: str,
    prompt: str,
    output_name: str,
    guidance: float = 3.0,
    edge_weight: float = 1.0,
) -> dict:
    """Create transfer configuration - edge control only for H100"""
    return {
        "name": output_name,
        "prompt": prompt,
        "video_path": video_path,
        "guidance": guidance,
        "edge": {
            "control_weight": edge_weight
        }
    }


def run_transfer(
    input_video: str,
    prompt: str,
    output_dir: str = "outputs",
    num_gpus: int = 1,
    guidance: float = 3.0,
    edge_weight: float = 1.0,
):
    input_video = os.path.abspath(input_video)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(input_video):
        print(f"ERROR: Input video not found: {input_video}")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_name = Path(input_video).stem

    config = create_transfer_config(
        video_path=input_video,
        prompt=prompt,
        output_name=output_name,
        guidance=guidance,
        edge_weight=edge_weight,
    )

    config_path = output_path / f"{output_name}_transfer_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print("NVIDIA Cosmos Transfer 2.5 - H100")
    print("=" * 70)
    print(f"Input Video: {input_video}")
    print(f"Prompt: {prompt}")
    print(f"Output Directory: {output_dir}")
    print(f"GPUs: {num_gpus}")
    print(f"Guidance Scale: {guidance}")
    print(f"Edge Weight: {edge_weight}")
    print("=" * 70)

    transfer_dir = Path("cosmos-transfer2.5")
    if not transfer_dir.exists():
        print("ERROR: cosmos-transfer2.5 directory not found.")
        print("Please run 'pixi run setup' first.")
        sys.exit(1)

    # Prepare environment with venv activation
    venv_python = transfer_dir / "venv" / "bin" / "python"
    if not venv_python.exists():
        print(f"ERROR: Virtual environment not found at {venv_python}")
        print("Please run 'bash setup.sh' first.")
        sys.exit(1)

    if num_gpus > 1:
        cmd = [
            str(venv_python),
            "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_port=12345",
            "-m", "examples.inference",
            "-i", str(config_path.absolute()),
            "-o", str(output_path.absolute()),
        ]
    else:
        cmd = [
            str(venv_python),
            "examples/inference.py",
            "-i", str(config_path.absolute()),
            "-o", str(output_path.absolute()),
        ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(
            cmd,
            cwd=transfer_dir,
            check=True,
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        )
        print()
        print("=" * 70)
        print(f"✓ Transfer complete!")
        print(f"  Output: {output_path.absolute()}")
        print("=" * 70)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Inference failed with exit code {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Cosmos Transfer 2.5 - H100")
    parser.add_argument("-i", "--input", required=True, help="Input video")
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt")
    parser.add_argument("-o", "--output", default="outputs", help="Output directory")
    parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--guidance", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--edge-weight", type=float, default=1.0, help="Edge weight")

    args = parser.parse_args()

    run_transfer(
        input_video=args.input,
        prompt=args.prompt,
        output_dir=args.output,
        num_gpus=args.gpus,
        guidance=args.guidance,
        edge_weight=args.edge_weight,
    )


if __name__ == "__main__":
    main()
