#!/usr/bin/env python3
"""
Batch process videos through Cosmos-Transfer2.5

This script processes all videos in the inputs directory through Cosmos-Transfer2.5,
generating photorealistic augmented versions.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

def validate_json_config(json_path: Path) -> bool:
    """
    Validate JSON configuration file.

    Args:
        json_path: Path to JSON file

    Returns:
        True if valid, False otherwise
    """
    try:
        with open(json_path) as f:
            config = json.load(f)

        # Check required fields
        required_fields = ["name", "prompt"]
        for field in required_fields:
            if field not in config:
                print(f"  ERROR: Missing required field '{field}' in {json_path.name}")
                return False

        # Validate guidance range
        if "guidance" in config and not (0 <= config["guidance"] <= 7):
            print(f"  WARNING: Guidance value {config['guidance']} outside range 0-7")

        return True
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON in {json_path.name}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: Failed to validate {json_path.name}: {e}")
        return False


def check_environment():
    """Check if Cosmos-Transfer environment is properly set up."""
    # Check if we're in the root directory
    script_dir = Path(__file__).parent
    cosmos_dir = script_dir / "cosmos-transfer" / "cosmos-transfer2.5"

    if not cosmos_dir.exists():
        print("ERROR: cosmos-transfer/cosmos-transfer2.5 directory not found!")
        print(f"Expected at: {cosmos_dir}")
        return False

    # Check if inference script exists
    inference_script = cosmos_dir / "examples" / "inference.py"
    if not inference_script.exists():
        print(f"ERROR: Inference script not found: {inference_script}")
        return False

    # Check activation script
    activate_script = script_dir / "cosmos-transfer" / "activate.sh"
    if not activate_script.exists():
        print(f"ERROR: Activation script not found: {activate_script}")
        return False

    return True


def run_cosmos_inference(
    json_path: Path,
    video_path: Path,
    output_dir: Path,
    cosmos_dir: Path,
    use_gpu: bool = True,
) -> tuple[bool, str]:
    """
    Run Cosmos-Transfer2.5 inference on a single video.

    Args:
        json_path: Path to JSON configuration
        video_path: Path to input video
        output_dir: Output directory for this video
        cosmos_dir: Path to cosmos-transfer2.5 directory
        use_gpu: Whether to use GPU

    Returns:
        Tuple of (success, output_path or error_message)
    """
    # Prepare paths
    cosmos_transfer_dir = cosmos_dir.parent  # cosmos-transfer directory
    inference_script = cosmos_dir / "examples" / "inference.py"
    venv_python = cosmos_dir / "venv" / "bin" / "python"

    # Verify venv exists
    if not venv_python.exists():
        return False, f"Virtual environment not found at {venv_python}. Run setup first."

    # Create temporary output directory
    temp_output = output_dir / "temp"
    temp_output.mkdir(parents=True, exist_ok=True)

    # Read JSON config to add video_path
    with open(json_path) as f:
        config = json.load(f)

    # Add video_path to config
    config["video_path"] = str(video_path.absolute())

    # Write temporary config
    temp_json = temp_output / json_path.name
    with open(temp_json, 'w') as f:
        json.dump(config, f, indent=2)

    # Build command using venv python directly
    cmd = f"""
    export PYTHONPATH="{cosmos_dir}:$PYTHONPATH" && \
    export HF_HOME="${{HF_HOME:-$HOME/.cache/huggingface}}" && \
    cd {cosmos_dir} && \
    {venv_python} examples/inference.py -i {temp_json.absolute()} -o {temp_output.absolute()}
    """

    print(f"  Running inference...")
    print(f"  Command: {cmd}")

    try:
        # Run inference with real-time output
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            error_msg = f"Inference failed with return code {result.returncode}"
            return False, error_msg

        # Find generated video
        # Cosmos-Transfer outputs: {name}.mp4 (augmented) and {name}_control_edge.mp4 (edge viz)
        # We want the augmented video, not the edge control
        video_name = config["name"]
        generated_video = temp_output / f"{video_name}.mp4"

        if not generated_video.exists():
            return False, f"Generated video not found at {generated_video}"

        # Verify we're not accidentally grabbing the edge control video
        if "_control_edge" in generated_video.name:
            return False, f"Found edge control instead of augmented video: {generated_video}"

        return True, str(generated_video)

    except subprocess.TimeoutExpired:
        return False, "Inference timed out after 1 hour"
    except Exception as e:
        return False, f"Exception during inference: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos through Cosmos-Transfer2.5"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("inputs"),
        help="Input directory containing videos and JSON configs (default: inputs)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip JSON validation"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing even if a video fails"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("COSMOS-TRANSFER BATCH PROCESSOR")
    print("=" * 80)
    print()

    # Check environment
    print("Checking environment...")
    if not check_environment():
        sys.exit(1)
    print("✓ Environment check passed\n")

    # Validate directories
    if not args.input_dir.exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find video+JSON pairs
    video_files = list(args.input_dir.glob("*.mp4")) + list(args.input_dir.glob("*.MP4"))

    if not video_files:
        print(f"No MP4 files found in {args.input_dir}")
        sys.exit(0)

    # Filter to only videos with corresponding JSON
    video_json_pairs = []
    for video_path in video_files:
        json_path = args.input_dir / f"{video_path.stem}.json"
        if json_path.exists():
            video_json_pairs.append((video_path, json_path))
        else:
            print(f"⊘ Skipping {video_path.name} (no JSON config)")

    if not video_json_pairs:
        print("\nNo videos with JSON configurations found!")
        print("\nCreate JSON configuration files for your videos in inputs/")
        print("See inputs/TEMPLATE.json for an example")
        sys.exit(1)

    print(f"Found {len(video_json_pairs)} video(s) with JSON configs:")
    for video_path, json_path in video_json_pairs:
        print(f"  - {video_path.name} + {json_path.name}")
    print()

    # Validate JSON configs
    if not args.skip_validation:
        print("Validating JSON configurations...")
        all_valid = True
        for video_path, json_path in video_json_pairs:
            print(f"  Checking {json_path.name}...")
            if not validate_json_config(json_path):
                all_valid = False

        if not all_valid:
            print("\nERROR: Some JSON files are invalid. Fix them and try again.")
            print("Or use --skip-validation to proceed anyway (not recommended).")
            sys.exit(1)
        print("✓ All JSON files valid\n")

    # Get cosmos directory
    cosmos_dir = Path(__file__).parent / "cosmos-transfer" / "cosmos-transfer2.5"

    # Process each video
    results = []
    for i, (video_path, json_path) in enumerate(video_json_pairs):
        video_name = video_path.stem
        print("=" * 80)
        print(f"Processing {i+1}/{len(video_json_pairs)}: {video_name}")
        print("=" * 80)

        # Create output subdirectory
        video_output_dir = args.output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Run inference
        success, result_path_or_error = run_cosmos_inference(
            json_path=json_path,
            video_path=video_path,
            output_dir=video_output_dir,
            cosmos_dir=cosmos_dir,
        )

        elapsed_time = time.time() - start_time

        if success:
            # Move output to final location
            final_output = video_output_dir / f"{video_name}_aug.mp4"
            shutil.move(result_path_or_error, final_output)

            # Clean up temp directory
            temp_dir = video_output_dir / "temp"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            print(f"✓ Success! Generated: {final_output}")
            print(f"  Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
            results.append(("success", video_name, elapsed_time))
        else:
            print(f"✗ Failed: {result_path_or_error}")
            results.append(("failed", video_name, result_path_or_error))

            if not args.continue_on_error:
                print("\nStopping due to error. Use --continue-on-error to process remaining videos.")
                break

        print()

    # Print summary
    print("=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)

    success_count = sum(1 for r in results if r[0] == "success")
    failed_count = sum(1 for r in results if r[0] == "failed")

    print(f"\nSuccessful: {success_count}/{len(video_json_pairs)}")
    print(f"Failed: {failed_count}/{len(video_json_pairs)}")

    if success_count > 0:
        print("\nSuccessful videos:")
        total_time = 0
        for status, name, time_or_msg in results:
            if status == "success":
                print(f"  ✓ {name} ({time_or_msg:.1f}s)")
                total_time += time_or_msg
        avg_time = total_time / success_count
        print(f"\nAverage time: {avg_time:.1f} seconds ({avg_time/60:.1f} minutes)")

    if failed_count > 0:
        print("\nFailed videos:")
        for status, name, time_or_msg in results:
            if status == "failed":
                print(f"  ✗ {name}")
                print(f"    Error: {time_or_msg[:200]}...")

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
