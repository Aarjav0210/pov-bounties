#!/usr/bin/env python3
"""
Batch process videos in the inputs directory using Video-Depth-Anything.

This script sequentially processes all video files in the inputs directory
and outputs depth videos to the outputs directory.
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

# Configuration
INPUTS_DIR = "/home/jay/Documents/Brown/Research/Cosmos/depth/inputs"
OUTPUTS_DIR = "/home/jay/Documents/Brown/Research/Cosmos/depth/outputs"
VIDEO_DEPTH_DIR = "/home/jay/Documents/Brown/Research/Cosmos/depth/Video-Depth-Anything"
PYTHON_PATH = "/home/jay/Documents/Brown/Research/Cosmos/depth/.pixi/envs/default/bin/python"

# Model settings (can be changed to 'vits', 'vitb', or 'vitl')
ENCODER = "vitl"  # Large model for best quality, requires ~23GB VRAM (H100 has plenty)

# Supported video extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']


def create_output_dir():
    """Create the outputs directory if it doesn't exist."""
    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUTS_DIR}")


def find_videos(input_dir):
    """Find all video files in the input directory."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    return sorted(videos)


def process_video(video_path, output_dir):
    """Process a single video file using Video-Depth-Anything."""
    video_name = os.path.basename(video_path)
    print(f"\n{'='*80}")
    print(f"Processing: {video_name}")
    print(f"{'='*80}")

    # Build command
    cmd = [
        PYTHON_PATH,
        "run.py",
        "--input_video", video_path,
        "--output_dir", output_dir,
        "--encoder", ENCODER
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {VIDEO_DEPTH_DIR}\n")

    # Run the depth processing
    try:
        result = subprocess.run(
            cmd,
            cwd=VIDEO_DEPTH_DIR,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        print(f"\n✓ Successfully processed: {video_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error processing {video_name}: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error processing {video_name}: {e}")
        return False


def main():
    """Main batch processing function."""
    print("Video-Depth-Anything Batch Processor")
    print("=" * 80)
    print(f"Input directory: {INPUTS_DIR}")
    print(f"Output directory: {OUTPUTS_DIR}")
    print(f"Encoder model: {ENCODER}")
    print("=" * 80)

    # Create output directory
    create_output_dir()

    # Find all videos
    videos = find_videos(INPUTS_DIR)

    if not videos:
        print(f"\n✗ No video files found in {INPUTS_DIR}")
        print(f"Looking for extensions: {', '.join(VIDEO_EXTENSIONS)}")
        return 1

    print(f"\nFound {len(videos)} video(s) to process:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {os.path.basename(video)}")

    # Process each video
    successful = 0
    failed = 0

    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing video...")

        if process_video(video_path, OUTPUTS_DIR):
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total videos: {len(videos)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nOutputs saved to: {OUTPUTS_DIR}")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
