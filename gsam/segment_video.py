#!/usr/bin/env python3
"""
Grounded-SAM Video Segmentation
Segment objects in video using text prompts
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
import supervision as sv
from PIL import Image

# Grounding DINO imports
from groundingdino.util.inference import Model as GroundingDINOModel

# Segment Anything imports
from segment_anything import sam_model_registry, SamPredictor


def load_models(
    grounding_dino_config: str,
    grounding_dino_checkpoint: str,
    sam_checkpoint: str,
    sam_version: str = "vit_h",
    device: str = "cuda:0",
) -> Tuple[GroundingDINOModel, SamPredictor]:
    """Load Grounding DINO and SAM models"""

    print(f"Loading models on {device}...")

    # Load Grounding DINO
    grounding_dino = GroundingDINOModel(
        model_config_path=grounding_dino_config,
        model_checkpoint_path=grounding_dino_checkpoint,
        device=device,
    )
    print("✓ Grounding DINO loaded")

    # Load SAM
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print("✓ SAM loaded")

    return grounding_dino, sam_predictor


def segment_frame(
    frame: np.ndarray,
    grounding_dino: GroundingDINOModel,
    sam_predictor: SamPredictor,
    text_prompt: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> Tuple[np.ndarray, sv.Detections]:
    """Segment a single frame using Grounded-SAM"""

    # Detect with Grounding DINO (expects BGR image)
    detections = grounding_dino.predict_with_classes(
        image=frame,
        classes=[text_prompt],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # Convert BGR to RGB for SAM
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # If no detections, return empty
    if len(detections.xyxy) == 0:
        return frame, detections

    # Segment with SAM
    sam_predictor.set_image(frame_rgb)

    # Convert boxes to SAM format
    boxes = detections.xyxy
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        torch.tensor(boxes, device=sam_predictor.device),
        frame_rgb.shape[:2]
    )

    # Predict masks
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Detach masks and convert to numpy
    masks_np = masks.cpu().numpy().squeeze(1)

    # Update detections with masks
    detections.mask = masks_np

    # Annotate frame manually with OpenCV (more robust than supervision annotators)
    annotated_frame = frame.copy()

    for i, (box, mask) in enumerate(zip(detections.xyxy, masks_np)):
        # Draw mask overlay
        color = np.array([30, 144, 255]) if i % 2 == 0 else np.array([255, 144, 30])  # Alternate colors
        mask_overlay = np.zeros_like(annotated_frame)
        mask_overlay[mask] = color
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask_overlay, 0.4, 0)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color.tolist(), 2)

        # Draw label
        label = f"obj_{i}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

    return annotated_frame, detections


def process_video(
    input_path: str,
    text_prompt: str,
    output_dir: str,
    grounding_dino: GroundingDINOModel,
    sam_predictor: SamPredictor,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
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
    print(f"Text prompt: '{text_prompt}'")
    print(f"Box threshold: {box_threshold}")
    print(f"Text threshold: {text_threshold}")

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
    all_detections = []

    print("\nProcessing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Segment frame
        annotated_frame, detections = segment_frame(
            frame,
            grounding_dino,
            sam_predictor,
            text_prompt,
            box_threshold,
            text_threshold,
        )

        # Save annotated frame to video
        out_video.write(annotated_frame)

        # Save masks if requested
        if save_masks and len(detections.xyxy) > 0:
            frame_masks_dir = masks_dir / f"frame_{frame_idx:04d}"
            frame_masks_dir.mkdir(exist_ok=True)

            for i, mask in enumerate(detections.mask):
                mask_img = (mask * 255).astype(np.uint8)
                mask_path = frame_masks_dir / f"mask_{i:02d}.png"
                cv2.imwrite(str(mask_path), mask_img)

        # Store detection info
        detection_info = {
            "frame": frame_idx,
            "num_detections": len(detections.xyxy),
            "boxes": detections.xyxy.tolist() if len(detections.xyxy) > 0 else [],
        }
        all_detections.append(detection_info)

        # Progress
        if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
            print(f"  Frame {frame_idx + 1}/{total_frames} - {len(detections.xyxy)} detections")

        frame_idx += 1

        # Clear GPU cache periodically
        if frame_idx % 30 == 0:
            torch.cuda.empty_cache()

    # Cleanup
    cap.release()
    out_video.release()

    # Save detection metadata
    metadata_path = output_dir / "detections.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "video": str(input_path),
            "text_prompt": text_prompt,
            "total_frames": frame_idx,
            "detections": all_detections,
        }, f, indent=2)

    print(f"\n✓ Processing complete!")
    print(f"  Annotated video: {output_video_path}")
    print(f"  Detections metadata: {metadata_path}")
    if save_masks:
        print(f"  Individual masks: {masks_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Grounded-SAM Video Segmentation")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-p", "--prompt", required=True, help="Text prompt (e.g. 'hand . object . table')")
    parser.add_argument("-o", "--output", default="outputs", help="Output directory")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text confidence threshold")
    parser.add_argument("--no-masks", action="store_true", help="Skip saving individual masks")
    parser.add_argument("--device", default="cuda:0", help="Device (cuda:0, cuda:1, etc)")

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)

    # Model paths
    # Get config from installed groundingdino package
    import groundingdino
    pkg_path = os.path.dirname(groundingdino.__file__)
    grounding_dino_config = os.path.join(pkg_path, "config", "GroundingDINO_SwinT_OGC.py")
    grounding_dino_checkpoint = "weights/groundingdino_swint_ogc.pth"
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"

    # Check if checkpoints exist
    for path in [grounding_dino_checkpoint, sam_checkpoint]:
        if not os.path.exists(path):
            print(f"ERROR: Model checkpoint not found: {path}")
            print("Please run the setup script first")
            sys.exit(1)

    print("=" * 70)
    print("Grounded-SAM Video Segmentation")
    print("=" * 70)

    # Load models
    grounding_dino, sam_predictor = load_models(
        grounding_dino_config,
        grounding_dino_checkpoint,
        sam_checkpoint,
        device=args.device,
    )

    # Process video
    process_video(
        args.input,
        args.prompt,
        args.output,
        grounding_dino,
        sam_predictor,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        save_masks=not args.no_masks,
    )


if __name__ == "__main__":
    main()
