#!/usr/bin/env python3
"""
Process video frames and estimate hand meshes with WiLoR
"""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add WiLoR to path
sys.path.insert(0, str(Path(__file__).parent / "WiLoR"))

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

# MANO hand skeleton connections (21 keypoints)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]


def get_all_frames(video_path: str):
    """Get all frames from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
        else:
            print(f"Warning: Could not read frame {idx}")
            break

    cap.release()
    return frames


def sample_frames(video_path: str, num_frames: int = 10, seed: int = 42):
    """Sample random frames from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    # Sample random frame indices
    random.seed(seed)
    if num_frames > total_frames:
        print(f"Warning: Requested {num_frames} frames but video only has {total_frames}")
        num_frames = total_frames

    frame_indices = sorted(random.sample(range(total_frames), num_frames))
    print(f"Sampled frame indices: {frame_indices}")

    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
        else:
            print(f"Warning: Could not read frame {idx}")

    cap.release()
    return frames


def project_keypoints(keypoints_3d, cam_trans, focal_length, img_res):
    """Project 3D keypoints to 2D image coordinates"""
    camera_center = [img_res[1] / 2., img_res[0] / 2.]
    K = np.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]

    # Add camera translation
    keypoints_cam = keypoints_3d + cam_trans

    # Project to 2D
    keypoints_2d_homo = K @ keypoints_cam.T
    keypoints_2d = (keypoints_2d_homo[:2] / keypoints_2d_homo[2:3]).T

    return keypoints_2d


def draw_skeleton(img, keypoints_2d, connections, color=(0, 255, 0), thickness=2):
    """Draw hand skeleton on image"""
    img = img.copy()

    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d):
            pt1 = tuple(keypoints_2d[start_idx].astype(int))
            pt2 = tuple(keypoints_2d[end_idx].astype(int))
            cv2.line(img, pt1, pt2, color, thickness)

    # Draw keypoints
    for kpt in keypoints_2d:
        pt = tuple(kpt.astype(int))
        cv2.circle(img, pt, 3, color, -1)

    return img


def process_frame(
    img_cv2: np.ndarray,
    frame_idx: int,
    model,
    model_cfg,
    detector,
    renderer,
    output_dir: Path,
    rescale_factor: float = 2.0,
    show_skeleton: bool = False,
    save_original: bool = False,
    device: str = "cuda"
):
    """Process a single frame with WiLoR"""

    # Detect hands
    detections = detector(img_cv2, conf=0.3, verbose=False)[0]
    bboxes = []
    is_right = []

    for det in detections:
        Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(Bbox[:4].tolist())

    if len(bboxes) == 0:
        return None  # Skip frames with no hands

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    # Prepare dataset
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_verts = []
    all_cam_t = []
    all_right = []
    all_joints_3d = []

    for batch in dataloader:
        batch = recursive_to(batch, device)

        with torch.no_grad():
            out = model(batch)

        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        # Process each detection
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
            is_right_hand = batch['right'][n].cpu().numpy()
            verts[:, 0] = (2 * is_right_hand - 1) * verts[:, 0]
            joints[:, 0] = (2 * is_right_hand - 1) * joints[:, 0]
            cam_t = pred_cam_t_full[n]

            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right_hand)
            all_joints_3d.append(joints)

    # Render overlay
    if len(all_verts) > 0:
        misc_args = dict(
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        cam_view = renderer.render_rgba_multiple(
            all_verts,
            cam_t=all_cam_t,
            render_res=img_size[0],
            is_right=all_right,
            **misc_args
        )

        # Overlay on original image
        input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
        input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

        # Convert to BGR for OpenCV
        result_img = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)

        # Draw skeleton if requested
        if show_skeleton:
            for joints_3d, cam_t in zip(all_joints_3d, all_cam_t):
                keypoints_2d = project_keypoints(joints_3d, cam_t, scaled_focal_length, img_size[0].cpu().numpy())
                result_img = draw_skeleton(result_img, keypoints_2d, HAND_CONNECTIONS, color=(0, 255, 0), thickness=2)

        # Save results
        output_path = output_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(output_path), result_img)

        # Save original if requested
        if save_original:
            orig_path = output_dir / f"frame_{frame_idx:05d}_original.jpg"
            cv2.imwrite(str(orig_path), img_cv2)

        return output_path.name
    return None


def process_video(
    video_path: str,
    output_dir: str,
    all_frames: bool = False,
    num_frames: int = 10,
    rescale_factor: float = 2.0,
    show_skeleton: bool = False,
    save_original: bool = False,
    seed: int = 42
):
    """Main processing function"""

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WiLoR Hand Estimation on Video")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'All frames' if all_frames else f'{num_frames} random frames'}")
    print(f"Show skeleton: {show_skeleton}")
    print()

    # Get frames
    if all_frames:
        print("Extracting all frames from video...")
        frames = get_all_frames(str(video_path))
    else:
        print(f"Sampling {num_frames} frames from video...")
        frames = sample_frames(str(video_path), num_frames, seed)
    print(f"Extracted {len(frames)} frames\n")

    # Load models
    print("Loading WiLoR model...")
    model_dir = Path(__file__).parent / "WiLoR" / "pretrained_models"
    model, model_cfg = load_wilor(
        checkpoint_path=str(model_dir / "wilor_final.ckpt"),
        cfg_path=str(model_dir / "model_config.yaml")
    )
    # Patch torch.load for YOLO compatibility with PyTorch 2.9+
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load

    detector = YOLO(str(model_dir / "detector.pt"))

    # Restore original torch.load
    torch.load = original_torch_load

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    detector = detector.to(device)
    model.eval()
    print(f"✓ Models loaded on {device}\n")

    # Setup renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Process frames
    print("Processing frames...")
    processed_count = 0
    skipped_count = 0

    for i, (frame_idx, frame) in enumerate(frames):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(frames)} frames...")

        result = process_frame(
            frame,
            frame_idx,
            model,
            model_cfg,
            detector,
            renderer,
            output_dir,
            rescale_factor=rescale_factor,
            show_skeleton=show_skeleton,
            save_original=save_original,
            device=device
        )

        if result:
            processed_count += 1
        else:
            skipped_count += 1

    print(f"\n{'='*70}")
    print("✓ Processing complete!")
    print(f"Processed: {processed_count} frames")
    print(f"Skipped (no hands): {skipped_count} frames")
    print(f"Output directory: {output_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Process video frames and estimate hand meshes with WiLoR"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "-o", "--output",
        default="hand_frames_output",
        help="Output directory (default: hand_frames_output)"
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Process all frames in the video (default: sample random frames)"
    )
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=10,
        help="Number of random frames to sample when not using --all-frames (default: 10)"
    )
    parser.add_argument(
        "--rescale-factor",
        type=float,
        default=2.0,
        help="Factor for padding the hand bounding box (default: 2.0)"
    )
    parser.add_argument(
        "--show-skeleton",
        action="store_true",
        help="Draw hand skeleton (21 keypoints) on overlay"
    )
    parser.add_argument(
        "--save-original",
        action="store_true",
        help="Also save original frames alongside overlays"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for frame sampling (default: 42)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)

    process_video(
        args.input,
        args.output,
        all_frames=args.all_frames,
        num_frames=args.num_frames,
        rescale_factor=args.rescale_factor,
        show_skeleton=args.show_skeleton,
        save_original=args.save_original,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
