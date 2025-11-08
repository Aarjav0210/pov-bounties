#!/usr/bin/env python3
"""
Automatic SAM Video Segmentation
Segment everything in video without text prompts
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# Segment Anything imports
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def load_sam(
    sam_checkpoint: str,
    sam_version: str = "vit_h",
    device: str = "cuda:0",
):
    """Load SAM model with automatic mask generator"""

    print(f"Loading SAM on {device}...")
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Create automatic mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,  # Grid points for mask generation
        pred_iou_thresh=0.86,  # Quality threshold
        stability_score_thresh=0.92,  # Stability threshold
        crop_n_layers=1,  # Crop layers for better segmentation
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Minimum mask size in pixels
    )

    print("✓ SAM loaded with automatic mask generation")
    return mask_generator


def segment_frame(
    frame: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator,
) -> Tuple[np.ndarray, List[dict]]:
    """Automatically segment a single frame"""

    # Convert BGR to RGB for SAM
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Generate masks automatically
    masks = mask_generator.generate(frame_rgb)

    # Sort by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Annotate frame
    annotated_frame = frame.copy()

    # Generate colors for each mask
    np.random.seed(42)  # Consistent colors
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        color = colors[i].tolist()

        # Draw mask overlay
        mask_overlay = np.zeros_like(annotated_frame)
        mask_overlay[mask] = color
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask_overlay, 0.35, 0)

        # Draw boundary
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated_frame, contours, -1, color, 2)

    return annotated_frame, masks


def process_video(
    input_path: str,
    output_dir: str,
    mask_generator: SamAutomaticMaskGenerator,
    save_masks: bool = True,
):
    """Process entire video frame by frame"""

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {input_path}")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    # Setup output video
    output_video_path = output_dir / f"{input_path.stem}_segmented.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Setup masks directory
    if save_masks:
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

    # Process frames
    frame_idx = 0
    all_segmentations = []

    print("\nProcessing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Segment frame
        annotated_frame, masks = segment_frame(frame, mask_generator)

        # Save annotated frame to video
        out_video.write(annotated_frame)

        # Save masks if requested
        if save_masks and len(masks) > 0:
            frame_masks_dir = masks_dir / f"frame_{frame_idx:04d}"
            frame_masks_dir.mkdir(exist_ok=True)

            for i, mask_data in enumerate(masks):
                mask = mask_data['segmentation']
                mask_img = (mask * 255).astype(np.uint8)
                mask_path = frame_masks_dir / f"mask_{i:02d}.png"
                cv2.imwrite(str(mask_path), mask_img)

        # Store segmentation info
        seg_info = {
            "frame": frame_idx,
            "num_masks": len(masks),
            "areas": [m['area'] for m in masks],
            "avg_stability": np.mean([m['stability_score'] for m in masks]) if masks else 0,
        }
        all_segmentations.append(seg_info)

        # Progress
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx + 1}/{total_frames} - {len(masks)} masks")

        frame_idx += 1

        # Clear GPU cache periodically
        if frame_idx % 30 == 0:
            torch.cuda.empty_cache()

    # Cleanup
    cap.release()
    out_video.release()

    # Save segmentation metadata
    metadata_path = output_dir / "segmentations.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "video": str(input_path),
            "total_frames": frame_idx,
            "segmentations": all_segmentations,
        }, f, indent=2)

    print(f"\n✓ Processing complete!")
    print(f"  Annotated video: {output_video_path}")
    print(f"  Segmentation metadata: {metadata_path}")
    if save_masks:
        print(f"  Individual masks: {masks_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Automatic SAM Video Segmentation")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", default="outputs_auto", help="Output directory")
    parser.add_argument("--no-masks", action="store_true", help="Skip saving individual masks")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cuda:1, etc)")

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)

    # Model paths
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"

    # Check if model exists
    if not os.path.exists(sam_checkpoint):
        print(f"ERROR: SAM checkpoint not found: {sam_checkpoint}")
        print("Please run 'pixi run setup' first")
        sys.exit(1)

    print("=" * 70)
    print("Automatic SAM Video Segmentation")
    print("=" * 70)

    # Load SAM
    mask_generator = load_sam(sam_checkpoint, device=args.device)

    # Process video
    process_video(
        args.input,
        args.output,
        mask_generator,
        save_masks=not args.no_masks,
    )


if __name__ == "__main__":
    main()
