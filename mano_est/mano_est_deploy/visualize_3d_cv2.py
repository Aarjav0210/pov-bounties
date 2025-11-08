#!/usr/bin/env python3
"""
Create 3D visualization video using opencv instead of matplotlib's FFMpegWriter
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


def visualize_3d_trajectory(data_file: str, output_video: str):
    """Create 3D visualization of hand trajectory using opencv"""

    print("\n" + "=" * 70)
    print("Creating 3D Visualization with OpenCV")
    print("=" * 70)

    # Load data
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    frames = data['frames']
    fps = data['fps']

    # Filter frames with hands
    frames_with_hands = [f for f in frames if len(f['hands']) > 0]
    print(f"Frames with hands: {len(frames_with_hands)}/{len(frames)}")

    if len(frames_with_hands) == 0:
        print("No hands detected in video!")
        return

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

    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize video writer (will set after first frame)
    video_writer = None

    print(f"Rendering {len(frames_with_hands)} frames...")

    for frame_num, frame_data in enumerate(frames_with_hands):
        if (frame_num + 1) % 50 == 0:
            print(f"  Progress: {frame_num + 1}/{len(frames_with_hands)} frames...")

        ax.cla()
        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Frame {frame_data["frame_idx"]} - 3D Hand Reconstruction')

        for hand_idx, hand in enumerate(frame_data['hands']):
            verts = hand['vertices']
            kpts = hand['keypoints']
            is_right = hand['is_right']

            color = 'blue' if is_right else 'red'

            # Plot mesh vertices as scatter
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

        # Set view angle
        ax.view_init(elev=20, azim=45)

        if frame_num == 0:
            ax.legend()

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
    print(f"✓ Video saved to: {output_video}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D hand trajectory visualization using OpenCV"
    )
    parser.add_argument(
        "data_file",
        help="Path to hand_data_3d.pkl file"
    )
    parser.add_argument(
        "-o", "--output",
        default="trajectory_3d.mp4",
        help="Output video file (default: trajectory_3d.mp4)"
    )

    args = parser.parse_args()

    visualize_3d_trajectory(args.data_file, args.output)


if __name__ == "__main__":
    main()
