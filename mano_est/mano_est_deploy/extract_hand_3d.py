#!/usr/bin/env python3
"""
Extract 3D hand data from video and create 3D visualizations
"""

import argparse
import os
import sys
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

# Add WiLoR to path
sys.path.insert(0, str(Path(__file__).parent / "WiLoR"))

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO

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
    return frames, fps


def extract_hand_data(
    video_path: str,
    output_dir: str,
    rescale_factor: float = 2.0,
    device: str = "cuda"
):
    """Extract 3D hand data from video"""

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Extracting 3D Hand Data")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print()

    # Get all frames
    print("Loading video frames...")
    frames, fps = get_all_frames(str(video_path))
    print(f"Loaded {len(frames)} frames\n")

    # Load models
    print("Loading WiLoR model...")
    model_dir = Path(__file__).parent / "WiLoR" / "pretrained_models"
    model, model_cfg = load_wilor(
        checkpoint_path=str(model_dir / "wilor_final.ckpt"),
        cfg_path=str(model_dir / "model_config.yaml")
    )

    # Patch torch.load for YOLO
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
    detector = YOLO(str(model_dir / "detector.pt"))
    torch.load = original_torch_load

    device_obj = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device_obj)
    detector = detector.to(device_obj)
    model.eval()
    print(f"✓ Models loaded on {device_obj}\n")

    # Extract data from all frames
    print("Processing frames...")
    all_frame_data = []

    for i, (frame_idx, frame) in enumerate(frames):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(frames)} frames...")

        # Detect hands
        detections = detector(frame, conf=0.3, verbose=False)[0]
        bboxes = []
        is_right = []

        for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())

        if len(bboxes) == 0:
            all_frame_data.append({
                'frame_idx': frame_idx,
                'hands': []
            })
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Prepare dataset
        dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        frame_hands = []

        for batch in dataloader:
            batch = recursive_to(batch, device_obj)

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

                # Store 3D data in camera space
                frame_hands.append({
                    'vertices': verts + cam_t,  # Transform to camera space
                    'keypoints': joints + cam_t,  # Transform to camera space
                    'is_right': bool(is_right_hand),
                    'cam_translation': cam_t
                })

        all_frame_data.append({
            'frame_idx': frame_idx,
            'hands': frame_hands
        })

    # Save data
    data_file = output_dir / 'hand_data_3d.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump({
            'frames': all_frame_data,
            'fps': fps,
            'video_path': str(video_path),
            'faces': model.mano.faces
        }, f)

    print(f"\n{'='*70}")
    print(f"✓ Extracted data from {len(all_frame_data)} frames")
    print(f"✓ Saved to: {data_file}")
    print("="*70)

    return data_file


def visualize_3d_trajectory(data_file: str, output_video: str = None):
    """Create 3D visualization of hand trajectory"""

    print("\n" + "=" * 70)
    print("Creating 3D Visualization")
    print("=" * 70)

    # Load data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    frames = data['frames']
    fps = data['fps']
    faces = data['faces']

    # Filter frames with hands
    frames_with_hands = [f for f in frames if len(f['hands']) > 0]
    print(f"Frames with hands: {len(frames_with_hands)}/{len(frames)}")

    if len(frames_with_hands) == 0:
        print("No hands detected in video!")
        return

    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Determine bounds from all data
    all_points = []
    for frame_data in frames_with_hands:
        for hand in frame_data['hands']:
            all_points.append(hand['vertices'])
            all_points.append(hand['keypoints'])

    all_points = np.concatenate(all_points, axis=0)

    # Set consistent axis limits
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    # Add padding
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
    ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
    ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    def update(frame_num):
        ax.cla()
        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        if frame_num >= len(frames_with_hands):
            return

        frame_data = frames_with_hands[frame_num]
        ax.set_title(f'Frame {frame_data["frame_idx"]} - 3D Hand Reconstruction')

        for hand_idx, hand in enumerate(frame_data['hands']):
            verts = hand['vertices']
            kpts = hand['keypoints']
            is_right = hand['is_right']

            color = 'blue' if is_right else 'red'

            # Plot mesh vertices as scatter
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                      c=color, alpha=0.1, s=1, label=f'{"Right" if is_right else "Left"} Hand Mesh')

            # Plot skeleton
            for start_idx, end_idx in HAND_CONNECTIONS:
                pts = np.array([kpts[start_idx], kpts[end_idx]])
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                       color='green', linewidth=2)

            # Plot keypoints
            ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2],
                      c='green', s=50, marker='o', edgecolors='black', linewidths=1)

        # Set view angle
        ax.view_init(elev=20, azim=45)

        if frame_num == 0:
            ax.legend()

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frames_with_hands),
                        interval=1000/fps, repeat=True)

    print(f"Saving video to: {output_video}")
    # Configure ffmpeg path explicitly for matplotlib
    import matplotlib as mpl
    mpl.rcParams['animation.ffmpeg_path'] = str(Path(__file__).parent / ".pixi/envs/default/bin/ffmpeg")
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_video, writer=writer, dpi=100)
    print(f"✓ Video saved!")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3D hand data and create trajectory visualizations"
    )
    parser.add_argument(
        "-i", "--input",
        required=False,
        help="Input video path (required unless using --visualize-only)"
    )
    parser.add_argument(
        "-o", "--output",
        default="hand_3d_output",
        help="Output directory (default: hand_3d_output)"
    )
    parser.add_argument(
        "--rescale-factor",
        type=float,
        default=2.0,
        help="Factor for padding the hand bounding box (default: 2.0)"
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract data, don't create visualization"
    )
    parser.add_argument(
        "--visualize-only",
        type=str,
        metavar="DATA_FILE",
        help="Only create visualization from existing data file"
    )
    parser.add_argument(
        "--output-video",
        type=str,
        help="Output video file (e.g., trajectory.mp4). If not specified, defaults to OUTPUT_DIR/trajectory_3d.mp4"
    )

    args = parser.parse_args()

    if args.visualize_only:
        # Just visualize existing data
        output_video = args.output_video
        if output_video is None:
            output_video = str(Path(args.visualize_only).parent / "trajectory_3d.mp4")
        visualize_3d_trajectory(args.visualize_only, output_video)
    else:
        # Extract data
        if not args.input:
            print(f"ERROR: --input is required when not using --visualize-only")
            sys.exit(1)

        if not os.path.exists(args.input):
            print(f"ERROR: Input video not found: {args.input}")
            sys.exit(1)

        data_file = extract_hand_data(
            args.input,
            args.output,
            rescale_factor=args.rescale_factor
        )

        # Visualize unless extract-only
        if not args.extract_only:
            output_video = args.output_video
            if output_video is None:
                output_video = str(Path(args.output) / "trajectory_3d.mp4")
            visualize_3d_trajectory(data_file, output_video)


if __name__ == "__main__":
    main()
