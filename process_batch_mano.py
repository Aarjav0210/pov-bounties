#!/usr/bin/env python3
"""
Batch process multiple videos through generate_all_4_outputs.py

This script:
1. Reads all MP4 videos from ./inputs/
2. Processes each through mano_est/mano_est_deploy/generate_all_4_outputs.py
3. Saves outputs to ./outputs/<video_name>/
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Define paths
    root_dir = Path(__file__).parent
    inputs_dir = root_dir / "inputs"
    outputs_dir = root_dir / "outputs"

    # Path to the generate_all_4_outputs.py script
    script_path = root_dir / "mano_est" / "mano_est_deploy" / "generate_all_4_outputs.py"
    activate_script = root_dir / "mano_est" / "mano_est_deploy" / "activate.sh"

    # Validate paths
    if not inputs_dir.exists():
        print(f"ERROR: Input directory not found: {inputs_dir}")
        print(f"Creating {inputs_dir}...")
        inputs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Please add MP4 videos to {inputs_dir} and run again.")
        sys.exit(1)

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        sys.exit(1)

    if not activate_script.exists():
        print(f"ERROR: Activate script not found: {activate_script}")
        sys.exit(1)

    # Create outputs directory
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Find all MP4 files in inputs
    video_files = list(inputs_dir.glob("*.mp4")) + list(inputs_dir.glob("*.MP4"))

    if len(video_files) == 0:
        print(f"No MP4 files found in {inputs_dir}")
        sys.exit(0)

    print("=" * 80)
    print("BATCH PROCESSING VIDEOS")
    print("=" * 80)
    print(f"Found {len(video_files)} video(s) to process:")
    for vf in video_files:
        print(f"  - {vf.name}")
    print()

    # Process each video
    for i, video_path in enumerate(video_files):
        video_name = video_path.stem  # filename without extension
        output_subdir = outputs_dir / video_name

        print("=" * 80)
        print(f"Processing video {i+1}/{len(video_files)}: {video_path.name}")
        print("=" * 80)
        print(f"Input: {video_path}")
        print(f"Output: {output_subdir}")
        print()

        # Create output subdirectory
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Change to the script directory
        script_dir = script_path.parent

        # Build command to run the script with source activate
        cmd = f"cd {script_dir} && source activate.sh && python generate_all_4_outputs.py -i {video_path} -o {output_subdir}"

        # Run the command
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                executable='/bin/bash',
                check=True,
                text=True,
                capture_output=False  # Show output in real-time
            )
            print(f"\n✓ Successfully processed {video_path.name}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ ERROR processing {video_path.name}")
            print(f"Return code: {e.returncode}")
            print("Continuing to next video...\n")
            continue

    print("=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {outputs_dir}")
    print("\nOutput structure:")
    print("  outputs/")
    for video_path in video_files:
        video_name = video_path.stem
        print(f"    {video_name}/")
        print(f"      1_mano_overlay.mp4")
        print(f"      2_skeleton_overlay.mp4")
        print(f"      3_mano_no_overlay.mp4")
        print(f"      4_skeleton_no_overlay.mp4")

if __name__ == "__main__":
    main()
