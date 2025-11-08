#!/usr/bin/env python3
"""
Create 3D visualization aligned with camera view from original video
"""

import argparse
import pickle
from pathlib import Path
import io

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# MANO hand skeleton connections (21 keypoints)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]


def fig_to_array(fig):
    """Convert matplotlib figure to numpy array"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img


def visualize_3d_camera_aligned(data_file: str, video_path: str, output_video: str, side_by_side: bool = False):
    """Create 3D visualization aligned with camera view"""

    print("\n" + "=" * 70)
    print("Creating Camera-Aligned 3D Visualization")
    print("=" * 70)

    # Load data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    frames_data = data['frames']
    fps = data['fps']

    # Filter frames with hands
    frames_with_hands = [(i, f) for i, f in enumerate(frames_data) if len(f['hands']) > 0]
    print(f"Frames with hands: {len(frames_with_hands)}/{len(frames_data)}")

    if len(frames_with_hands) == 0:
        print("No hands detected in video!")
        return

    # Load original video if side-by-side
    video_frames = {}
    if side_by_side:
        print("Loading original video frames...")
        cap = cv2.VideoCapture(video_path)
        for frame_idx, frame_data in frames_with_hands:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_data['frame_idx'])
            ret, frame = cap.read()
            if ret:
                video_frames[frame_data['frame_idx']] = frame
        cap.release()

    # Determine bounds from all data
    all_points = []
    for _, frame_data in frames_with_hands:
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

    # Setup figure
    if side_by_side:
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(122, projection='3d')
    else:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Initialize video writer
    video_writer = None

    print(f"Rendering {len(frames_with_hands)} frames...")

    for idx, (frame_idx, frame_data) in enumerate(frames_with_hands):
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{len(frames_with_hands)} frames...")

        # Clear 3D axis
        ax.cla()

        # Set limits - camera-aligned coordinate system
        # In camera coords: X is right, Y is down, Z is forward (depth)
        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])

        # Label axes in camera coordinate system
        ax.set_xlabel('X (m) - Right')
        ax.set_ylabel('Y (m) - Down')
        ax.set_zlabel('Z (m) - Depth')
        ax.set_title(f'Frame {frame_data["frame_idx"]} - Camera-Aligned 3D View')

        for hand in frame_data['hands']:
            verts = hand['vertices']
            kpts = hand['keypoints']
            is_right = hand['is_right']

            color = 'blue' if is_right else 'red'

            # Plot mesh vertices
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                      c=color, alpha=0.1, s=1)

            # Plot skeleton
            for start_idx, end_idx in HAND_CONNECTIONS:
                pts = np.array([kpts[start_idx], kpts[end_idx]])
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                       color='green', linewidth=2)

            # Plot keypoints
            ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2],
                      c='green', s=50, marker='o', edgecolors='black', linewidths=1)

        # Camera-aligned view: looking down the Z axis (from camera)
        # This matches the camera's perspective
        ax.view_init(elev=0, azim=-90)  # Look from camera position

        # Invert Y axis to match image coordinates (Y down in camera)
        ax.invert_yaxis()

        if idx == 0:
            ax.legend(['Right Hand' if frame_data['hands'][0]['is_right'] else 'Left Hand'])

        if side_by_side:
            # Add original video frame on left
            ax_video = fig.add_subplot(121)
            ax_video.clear()
            if frame_data['frame_idx'] in video_frames:
                frame = video_frames[frame_data['frame_idx']]
                ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax_video.set_title(f'Original Video - Frame {frame_data["frame_idx"]}')
            ax_video.axis('off')

        # Convert figure to image
        img = fig_to_array(fig)

        # Initialize video writer with first frame dimensions
        if video_writer is None:
            height, width = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            print(f"Video dimensions: {width}x{height}")

        # Write frame
        video_writer.write(img)

    # Cleanup
    video_writer.release()
    plt.close()

    print(f"\n{'='*70}")
    print(f"✓ Camera-aligned video saved to: {output_video}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Create camera-aligned 3D hand trajectory visualization"
    )
    parser.add_argument(
        "data_file",
        help="Path to hand_data_3d.pkl file"
    )
    parser.add_argument(
        "-v", "--video",
        help="Path to original video (required for side-by-side mode)"
    )
    parser.add_argument(
        "-o", "--output",
        default="trajectory_3d_aligned.mp4",
        help="Output video file (default: trajectory_3d_aligned.mp4)"
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Show original video and 3D view side-by-side"
    )

    args = parser.parse_args()

    if args.side_by_side and not args.video:
        print("ERROR: --video is required when using --side-by-side")
        return

    visualize_3d_camera_aligned(
        args.data_file,
        args.video,
        args.output,
        side_by_side=args.side_by_side
    )


if __name__ == "__main__":
    main()
