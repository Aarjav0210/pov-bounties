#!/usr/bin/env python3
"""
Full Pipeline Batch Processor

Processes videos through the complete pipeline:
1. Depth estimation (Video-Depth-Anything)
2. MANO hand estimation (WiLoR)
3. Object segmentation (Grounded-SAM)
4. Photorealistic augmentation (Cosmos-Transfer2.5)

Each video flows through all four stages sequentially, with outputs organized
in a structured directory per video.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


class PipelineProcessor:
    """Manages the full pipeline processing for videos."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        root_dir: Path,
        depth_encoder: str = "vits",
        gsam_prompt: str = "hand",
        continue_on_error: bool = False,
        skip_stages: list = None,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.depth_encoder = depth_encoder
        self.gsam_prompt = gsam_prompt
        self.continue_on_error = continue_on_error
        self.skip_stages = skip_stages or []

        # Define paths to each model
        self.depth_dir = root_dir / "depth"
        self.depth_python = self.depth_dir / ".pixi" / "envs" / "default" / "bin" / "python"
        self.depth_script = self.depth_dir / "Video-Depth-Anything" / "run.py"

        self.mano_dir = root_dir / "mano_est" / "mano_est_deploy"
        self.mano_script = self.mano_dir / "generate_all_4_outputs.py"
        self.mano_activate = self.mano_dir / "activate.sh"

        self.gsam_dir = root_dir / "gsam"
        self.gsam_python = self.gsam_dir / ".pixi" / "envs" / "default" / "bin" / "python"
        self.gsam_script = self.gsam_dir / "segment_video.py"
        self.gsam_activate = self.gsam_dir / "activate.sh"

        self.cosmos_dir = root_dir / "cosmos-transfer" / "cosmos-transfer2.5"
        self.cosmos_python = self.cosmos_dir / "venv" / "bin" / "python"
        self.cosmos_script = self.cosmos_dir / "examples" / "inference.py"

    def validate_environment(self) -> bool:
        """Validate that all required environments and scripts exist."""
        print("Validating environments...")

        if "depth" not in self.skip_stages:
            if not self.depth_python.exists():
                print(f"✗ Depth Python not found: {self.depth_python}")
                print("  Run: cd depth && pixi install")
                return False
            if not self.depth_script.exists():
                print(f"✗ Depth script not found: {self.depth_script}")
                return False
            print("  ✓ Depth environment OK")

        if "mano" not in self.skip_stages:
            if not self.mano_script.exists():
                print(f"✗ MANO script not found: {self.mano_script}")
                return False
            if not self.mano_activate.exists():
                print(f"✗ MANO activate script not found: {self.mano_activate}")
                return False
            print("  ✓ MANO environment OK")

        if "gsam" not in self.skip_stages:
            if not self.gsam_python.exists():
                print(f"✗ GSAM Python not found: {self.gsam_python}")
                print("  Run: cd gsam && pixi install")
                return False
            if not self.gsam_script.exists():
                print(f"✗ GSAM script not found: {self.gsam_script}")
                return False
            if not self.gsam_activate.exists():
                print(f"✗ GSAM activate script not found: {self.gsam_activate}")
                return False
            print("  ✓ GSAM environment OK")

        if "cosmos" not in self.skip_stages:
            if not self.cosmos_python.exists():
                print(f"✗ Cosmos Python not found: {self.cosmos_python}")
                print("  Run: cd cosmos-transfer && bash setup.sh")
                return False
            if not self.cosmos_script.exists():
                print(f"✗ Cosmos script not found: {self.cosmos_script}")
                return False
            print("  ✓ Cosmos environment OK")

        print("✓ All environments validated\n")
        return True

    def run_depth_estimation(
        self, video_path: Path, output_dir: Path
    ) -> Tuple[bool, str]:
        """Run depth estimation on a video."""
        print(f"\n{'='*80}")
        print("STAGE 1: DEPTH ESTIMATION")
        print(f"{'='*80}")

        depth_output = output_dir / "depth.mp4"

        # Build command
        cmd = [
            str(self.depth_python),
            "run.py",
            "--input_video", str(video_path.absolute()),
            "--output_dir", str(output_dir.absolute()),
            "--encoder", self.depth_encoder
        ]

        print(f"Encoder: {self.depth_encoder}")
        print(f"Command: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.depth_dir / "Video-Depth-Anything",
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr
            )

            # Video-Depth-Anything outputs: {video_name}_vis.mp4 (depth visualization)
            # Find the generated depth video
            video_stem = video_path.stem
            depth_video_path = output_dir / f"{video_stem}_vis.mp4"

            if depth_video_path.exists():
                # Rename to standardized name
                shutil.move(depth_video_path, depth_output)
                print(f"✓ Depth estimation complete: {depth_output.name}")
                return True, str(depth_output)
            else:
                # List what files were actually created for debugging
                created_files = list(output_dir.glob("*.mp4"))
                return False, f"Depth video not found. Expected: {depth_video_path.name}, Found: {[f.name for f in created_files]}"

        except subprocess.CalledProcessError as e:
            return False, f"Depth estimation failed with return code {e.returncode}"
        except Exception as e:
            return False, f"Depth estimation error: {str(e)}"

    def run_mano_estimation(
        self, video_path: Path, output_dir: Path
    ) -> Tuple[bool, str]:
        """Run MANO hand estimation on a video."""
        print(f"\n{'='*80}")
        print("STAGE 2: MANO HAND ESTIMATION")
        print(f"{'='*80}")

        # Build command using activate script
        cmd = f"cd {self.mano_dir} && source activate.sh && python generate_all_4_outputs.py -i {video_path.absolute()} -o {output_dir.absolute()}"

        print(f"Command: {cmd}\n")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                executable='/bin/bash',
                check=True
            )

            # Check that all 4 outputs were generated
            expected_outputs = [
                "1_mano_overlay.mp4",
                "2_skeleton_overlay.mp4",
                "3_mano_no_overlay.mp4",
                "4_skeleton_no_overlay.mp4"
            ]

            missing = []
            for output_name in expected_outputs:
                if not (output_dir / output_name).exists():
                    missing.append(output_name)

            if missing:
                return False, f"Missing outputs: {', '.join(missing)}"

            print(f"✓ MANO estimation complete: generated 4 outputs")
            return True, "All MANO outputs generated"

        except subprocess.CalledProcessError as e:
            return False, f"MANO estimation failed with return code {e.returncode}"
        except Exception as e:
            return False, f"MANO estimation error: {str(e)}"

    def run_gsam_segmentation(
        self, video_path: Path, output_dir: Path
    ) -> Tuple[bool, str]:
        """Run GSAM object segmentation on a video."""
        print(f"\n{'='*80}")
        print("STAGE 3: OBJECT SEGMENTATION (GSAM)")
        print(f"{'='*80}")

        # Build command using pixi python directly
        cmd = f"cd {self.gsam_dir} && source activate.sh && {self.gsam_python} segment_video.py -i {video_path.absolute()} -p '{self.gsam_prompt}' -o {output_dir.absolute()} --no-masks"

        print(f"Text prompt: {self.gsam_prompt}")
        print(f"Command: {cmd}\n")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                executable='/bin/bash',
                check=True
            )

            # Check that segmented video was generated
            # GSAM outputs: {video_stem}_segmented.mp4
            video_stem = video_path.stem
            segmented_video = output_dir / f"{video_stem}_segmented.mp4"
            detections_json = output_dir / "detections.json"

            if not segmented_video.exists():
                return False, f"Segmented video not found: {segmented_video.name}"

            print(f"✓ GSAM segmentation complete: {segmented_video.name}")
            if detections_json.exists():
                print(f"  Detections saved: {detections_json.name}")

            return True, str(segmented_video)

        except subprocess.CalledProcessError as e:
            return False, f"GSAM segmentation failed with return code {e.returncode}"
        except Exception as e:
            return False, f"GSAM segmentation error: {str(e)}"

    def run_cosmos_augmentation(
        self, video_path: Path, json_path: Path, output_dir: Path, video_name: str
    ) -> Tuple[bool, str]:
        """Run Cosmos-Transfer augmentation on a video."""
        print(f"\n{'='*80}")
        print("STAGE 4: COSMOS PHOTOREALISTIC AUGMENTATION")
        print(f"{'='*80}")

        if not json_path.exists():
            return False, f"JSON config not found: {json_path}"

        # Create temp directory for Cosmos output
        temp_output = output_dir / "temp_cosmos"
        temp_output.mkdir(parents=True, exist_ok=True)

        # Read and modify JSON config
        with open(json_path) as f:
            config = json.load(f)

        config["video_path"] = str(video_path.absolute())
        temp_json = temp_output / json_path.name

        with open(temp_json, 'w') as f:
            json.dump(config, f, indent=2)

        # Build command
        cmd = f"""
        export PYTHONPATH="{self.cosmos_dir}:$PYTHONPATH" && \
        export HF_HOME="${{HF_HOME:-$HOME/.cache/huggingface}}" && \
        cd {self.cosmos_dir} && \
        {self.cosmos_python} examples/inference.py -i {temp_json.absolute()} -o {temp_output.absolute()}
        """

        print(f"JSON config: {json_path.name}")
        print(f"Command: {cmd}\n")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                executable='/bin/bash',
                check=True,
                timeout=3600
            )

            # Find the augmented video (not the control edge video)
            config_name = config["name"]
            generated_video = temp_output / f"{config_name}.mp4"

            if not generated_video.exists():
                return False, f"Generated video not found: {generated_video}"

            # Move to final location
            final_output = output_dir / f"{video_name}_aug.mp4"
            shutil.move(generated_video, final_output)

            # Clean up temp directory
            shutil.rmtree(temp_output)

            print(f"✓ Cosmos augmentation complete: {final_output.name}")
            return True, str(final_output)

        except subprocess.TimeoutExpired:
            return False, "Cosmos augmentation timed out (1 hour limit)"
        except subprocess.CalledProcessError as e:
            return False, f"Cosmos augmentation failed with return code {e.returncode}"
        except Exception as e:
            return False, f"Cosmos augmentation error: {str(e)}"

    def process_video(
        self, video_path: Path, json_path: Optional[Path] = None
    ) -> dict:
        """Process a single video through the full pipeline."""
        video_name = video_path.stem
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"PROCESSING: {video_name}")
        print(f"{'='*80}")
        print(f"Input: {video_path}")
        print(f"Output: {video_output_dir}")

        results = {
            "video_name": video_name,
            "stages": {},
            "total_time": 0,
            "success": True
        }

        start_time = time.time()

        # Stage 1: Depth Estimation
        if "depth" not in self.skip_stages:
            stage_start = time.time()
            success, message = self.run_depth_estimation(video_path, video_output_dir)
            stage_time = time.time() - stage_start

            results["stages"]["depth"] = {
                "success": success,
                "message": message,
                "time": stage_time
            }

            if not success:
                print(f"\n✗ Depth estimation failed: {message}")
                results["success"] = False
                if not self.continue_on_error:
                    results["total_time"] = time.time() - start_time
                    return results
        else:
            print("\n⊘ Skipping depth estimation")

        # Stage 2: MANO Hand Estimation
        if "mano" not in self.skip_stages:
            stage_start = time.time()
            success, message = self.run_mano_estimation(video_path, video_output_dir)
            stage_time = time.time() - stage_start

            results["stages"]["mano"] = {
                "success": success,
                "message": message,
                "time": stage_time
            }

            if not success:
                print(f"\n✗ MANO estimation failed: {message}")
                results["success"] = False
                if not self.continue_on_error:
                    results["total_time"] = time.time() - start_time
                    return results
        else:
            print("\n⊘ Skipping MANO estimation")

        # Stage 3: GSAM Object Segmentation
        if "gsam" not in self.skip_stages:
            stage_start = time.time()
            success, message = self.run_gsam_segmentation(video_path, video_output_dir)
            stage_time = time.time() - stage_start

            results["stages"]["gsam"] = {
                "success": success,
                "message": message,
                "time": stage_time
            }

            if not success:
                print(f"\n✗ GSAM segmentation failed: {message}")
                results["success"] = False
                if not self.continue_on_error:
                    results["total_time"] = time.time() - start_time
                    return results
        else:
            print("\n⊘ Skipping GSAM segmentation")

        # Stage 4: Cosmos Augmentation
        if "cosmos" not in self.skip_stages:
            if json_path and json_path.exists():
                stage_start = time.time()
                success, message = self.run_cosmos_augmentation(
                    video_path, json_path, video_output_dir, video_name
                )
                stage_time = time.time() - stage_start

                results["stages"]["cosmos"] = {
                    "success": success,
                    "message": message,
                    "time": stage_time
                }

                if not success:
                    print(f"\n✗ Cosmos augmentation failed: {message}")
                    results["success"] = False
            else:
                print(f"\n⊘ Skipping Cosmos augmentation (no JSON config: {video_name}.json)")
                results["stages"]["cosmos"] = {
                    "success": False,
                    "message": "No JSON config",
                    "time": 0
                }
        else:
            print("\n⊘ Skipping Cosmos augmentation")

        results["total_time"] = time.time() - start_time

        if results["success"]:
            print(f"\n✓ Pipeline complete for {video_name}")
            print(f"  Total time: {results['total_time']:.1f}s ({results['total_time']/60:.1f} min)")

        return results

    def run_batch(self) -> list:
        """Run the pipeline on all videos in the input directory."""
        # Find all video files
        video_files = list(self.input_dir.glob("*.mp4")) + list(self.input_dir.glob("*.MP4"))

        if not video_files:
            print(f"No MP4 files found in {self.input_dir}")
            return []

        print(f"Found {len(video_files)} video(s) to process:")
        for vf in video_files:
            json_path = self.input_dir / f"{vf.stem}.json"
            has_json = "✓" if json_path.exists() else "✗"
            print(f"  - {vf.name} (JSON: {has_json})")
        print()

        # Process each video
        all_results = []
        for i, video_path in enumerate(video_files):
            json_path = self.input_dir / f"{video_path.stem}.json"

            print(f"\n{'#'*80}")
            print(f"VIDEO {i+1}/{len(video_files)}")
            print(f"{'#'*80}")

            results = self.process_video(video_path, json_path)
            all_results.append(results)

        return all_results


def print_summary(results: list):
    """Print a summary of all processing results."""
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")

    total_videos = len(results)
    successful_videos = sum(1 for r in results if r["success"])
    failed_videos = total_videos - successful_videos

    print(f"\nTotal videos: {total_videos}")
    print(f"Successful: {successful_videos}")
    print(f"Failed: {failed_videos}")

    # Stage-wise summary
    stages = ["depth", "mano", "gsam", "cosmos"]
    print(f"\n{'Stage':<15} {'Success':<10} {'Failed':<10} {'Skipped':<10}")
    print("-" * 45)

    for stage in stages:
        success = sum(1 for r in results if stage in r["stages"] and r["stages"][stage]["success"])
        failed = sum(1 for r in results if stage in r["stages"] and not r["stages"][stage]["success"])
        skipped = total_videos - success - failed
        print(f"{stage.capitalize():<15} {success:<10} {failed:<10} {skipped:<10}")

    # Per-video details
    print(f"\n{'Video':<30} {'Depth':<8} {'MANO':<8} {'GSAM':<8} {'Cosmos':<8} {'Time':<12}")
    print("-" * 90)

    for r in results:
        video_name = r["video_name"][:28]
        depth_status = "✓" if "depth" in r["stages"] and r["stages"]["depth"]["success"] else ("✗" if "depth" in r["stages"] else "-")
        mano_status = "✓" if "mano" in r["stages"] and r["stages"]["mano"]["success"] else ("✗" if "mano" in r["stages"] else "-")
        gsam_status = "✓" if "gsam" in r["stages"] and r["stages"]["gsam"]["success"] else ("✗" if "gsam" in r["stages"] else "-")
        cosmos_status = "✓" if "cosmos" in r["stages"] and r["stages"]["cosmos"]["success"] else ("✗" if "cosmos" in r["stages"] else "-")
        total_time = f"{r['total_time']:.1f}s"

        print(f"{video_name:<30} {depth_status:<8} {mano_status:<8} {gsam_status:<8} {cosmos_status:<8} {total_time:<12}")

    total_time = sum(r["total_time"] for r in results)
    print(f"\nTotal processing time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if successful_videos > 0:
        avg_time = total_time / successful_videos
        print(f"Average time per video: {avg_time:.1f}s ({avg_time/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(
        description="Process videos through the full pipeline: Depth → MANO → GSAM → Cosmos"
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
        "--depth-encoder",
        choices=["vits", "vitb", "vitl"],
        default="vitl",
        help="Depth encoder model: vits (small, 7GB VRAM), vitb (base), vitl (large, best quality, ~23GB VRAM)"
    )
    parser.add_argument(
        "--gsam-prompt",
        type=str,
        default="hand",
        help="Text prompt for GSAM object segmentation (default: 'hand')"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["depth", "mano", "gsam", "cosmos"],
        default=[],
        help="Skip specific pipeline stages"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing next video even if current one fails"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FULL PIPELINE BATCH PROCESSOR")
    print("=" * 80)
    print(f"Pipeline: Depth → MANO → GSAM → Cosmos")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Depth encoder: {args.depth_encoder}")
    print(f"GSAM prompt: {args.gsam_prompt}")
    if args.skip:
        print(f"Skipping stages: {', '.join(args.skip)}")
    print("=" * 80)
    print()

    # Validate directories
    if not args.input_dir.exists():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    root_dir = Path(__file__).parent
    processor = PipelineProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        root_dir=root_dir,
        depth_encoder=args.depth_encoder,
        gsam_prompt=args.gsam_prompt,
        continue_on_error=args.continue_on_error,
        skip_stages=args.skip,
    )

    # Validate environment
    if not processor.validate_environment():
        sys.exit(1)

    # Run batch processing
    results = processor.run_batch()

    # Print summary
    if results:
        print_summary(results)
    else:
        print("No videos processed")

    # Exit with error code if any videos failed
    failed_count = sum(1 for r in results if not r["success"])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
