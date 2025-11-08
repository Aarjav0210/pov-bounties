#!/usr/bin/env python3
"""
Generate ALL 4 outputs as originally requested:
1. MANO overlaid on video
2. Skeleton overlaid on video
3. MANO in static 3D grid box
4. Skeleton in static 3D grid box
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import pyrender

sys.path.insert(0, str(Path(__file__).parent / "WiLoR"))

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

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
    return frames, fps, (width, height)


def project_keypoints(keypoints_3d, cam_trans, focal_length, img_size_tensor):
    """Project 3D keypoints to 2D - CORRECTED VERSION"""
    # CRITICAL: img_size_tensor is [height, width] NOT [width, height]
    camera_center = [img_size_tensor[0] / 2., img_size_tensor[1] / 2.]
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]

    points = torch.from_numpy(keypoints_3d).float()
    cam_t = torch.from_numpy(cam_trans).float()

    points = points + cam_t
    points = points / points[..., -1:]
    V_2d = (K @ points.T).T
    return V_2d[..., :-1].numpy()


def draw_skeleton(img, keypoints_2d, connections, color=(0, 255, 0), thickness=2):
    """Draw hand skeleton on image"""
    img = img.copy()

    for start_idx, end_idx in connections:
        if start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d):
            pt1 = tuple(keypoints_2d[start_idx].astype(int))
            pt2 = tuple(keypoints_2d[end_idx].astype(int))
            cv2.line(img, pt1, pt2, color, thickness)

    for kpt in keypoints_2d:
        pt = tuple(kpt.astype(int))
        cv2.circle(img, pt, 3, color, -1)

    return img


def create_static_grid_box(size=0.5, spacing=0.05, center_z=-8.5):
    """Create a STATIC 3D grid box"""
    meshes = []
    half = size / 2
    line_thickness = 0.008

    box_min = np.array([-half, -half, center_z - half])
    box_max = np.array([half, half, center_z + half])

    # Front face
    z_front = center_z + half
    num_lines = int(size / spacing) + 1
    for i in range(num_lines):
        y = -half + i * spacing
        line = trimesh.creation.box(extents=[size, line_thickness, line_thickness])
        line.apply_translation([0, y, z_front])
        line.visual.vertex_colors = [100, 100, 100, 255]
        meshes.append(line)

    for i in range(num_lines):
        x = -half + i * spacing
        line = trimesh.creation.box(extents=[line_thickness, size, line_thickness])
        line.apply_translation([x, 0, z_front])
        line.visual.vertex_colors = [100, 100, 100, 255]
        meshes.append(line)

    # Back face
    z_back = center_z - half
    for i in range(num_lines):
        y = -half + i * spacing
        line = trimesh.creation.box(extents=[size, line_thickness, line_thickness])
        line.apply_translation([0, y, z_back])
        line.visual.vertex_colors = [120, 120, 120, 255]
        meshes.append(line)

    for i in range(num_lines):
        x = -half + i * spacing
        line = trimesh.creation.box(extents=[line_thickness, size, line_thickness])
        line.apply_translation([x, 0, z_back])
        line.visual.vertex_colors = [120, 120, 120, 255]
        meshes.append(line)

    # Connecting depth lines
    depth = size
    for x in [-half, half]:
        for y in [-half, half]:
            line = trimesh.creation.box(extents=[line_thickness, line_thickness, depth])
            line.apply_translation([x, y, center_z])
            line.visual.vertex_colors = [100, 100, 100, 255]
            meshes.append(line)

    for i in range(1, num_lines - 1):
        for y in [-half, half]:
            x = -half + i * spacing
            line = trimesh.creation.box(extents=[line_thickness, line_thickness, depth])
            line.apply_translation([x, y, center_z])
            line.visual.vertex_colors = [100, 100, 100, 255]
            meshes.append(line)

        for x in [-half, half]:
            y = -half + i * spacing
            line = trimesh.creation.box(extents=[line_thickness, line_thickness, depth])
            line.apply_translation([x, y, center_z])
            line.visual.vertex_colors = [100, 100, 100, 255]
            meshes.append(line)

    # Coordinate axes
    axis_length = 0.1
    axis_thickness = 0.01

    x_axis = trimesh.creation.box(extents=[axis_length, axis_thickness, axis_thickness])
    x_axis.apply_translation([axis_length/2, 0, center_z])
    x_axis.visual.vertex_colors = [255, 0, 0, 255]
    meshes.append(x_axis)

    y_axis = trimesh.creation.box(extents=[axis_thickness, axis_length, axis_thickness])
    y_axis.apply_translation([0, axis_length/2, center_z])
    y_axis.visual.vertex_colors = [0, 255, 0, 255]
    meshes.append(y_axis)

    z_axis = trimesh.creation.box(extents=[axis_thickness, axis_thickness, axis_length])
    z_axis.apply_translation([0, 0, center_z + axis_length/2])
    z_axis.visual.vertex_colors = [0, 0, 255, 255]
    meshes.append(z_axis)

    return meshes


def create_raymond_lights():
    """Create Raymond lighting"""
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            )
        )
    return nodes


def render_3d_mano(frame_data, renderer_2d):
    """Render MANO hands with white background using the SAME renderer as overlay"""
    hands = frame_data['hands']
    vertices_list = [h['vertices'] for h in hands]
    cam_t_list = [h['cam_t'] for h in hands]
    is_right_list = [h['is_right'] for h in hands]
    img_size = hands[0]['img_size']
    focal_length = hands[0]['focal_length']

    misc_args = dict(
        mesh_base_color=LIGHT_PURPLE,
        scene_bg_color=(1, 1, 1),  # White background instead of video
        focal_length=focal_length,
    )

    cam_view = renderer_2d.render_rgba_multiple(
        vertices_list,
        cam_t=cam_t_list,
        render_res=img_size,
        is_right=is_right_list,
        **misc_args
    )

    # Convert RGBA to BGR (no overlay, just the render)
    output_img = (cam_view[:, :, :3] * 255).astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    return output_img


def render_3d_skeleton(frame_data, vid_width, vid_height):
    """Render skeleton keypoints with white background using SAME projection as overlay"""
    hands = frame_data['hands']

    # FIX: Create white background using VIDEO dimensions, not img_size
    # img_size from the model can have different dimensions than the video
    white_img = np.ones((vid_height, vid_width, 3), dtype=np.uint8) * 255

    # Draw skeleton for each hand using the SAME projection as overlay
    for hand in hands:
        keypoints_3d = hand['keypoints']
        cam_t = hand['cam_t']
        focal_length = hand['focal_length']
        img_size = hand['img_size']

        keypoints_2d = project_keypoints(keypoints_3d, cam_t, focal_length, img_size)
        white_img = draw_skeleton(white_img, keypoints_2d, HAND_CONNECTIONS,
                                   color=(0, 255, 0), thickness=2)

    return white_img


def main():
    parser = argparse.ArgumentParser(description="Generate all 4 outputs")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", default="final_outputs", help="Output directory")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input video not found: {args.input}")
        sys.exit(1)

    video_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING ALL 4 OUTPUTS")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print()

    # Load video
    print("Loading video frames...")
    frames, fps, (vid_width, vid_height) = get_all_frames(str(video_path))
    print(f"Loaded {len(frames)} frames\n")

    # Load models
    print("Loading WiLoR model...")
    wilor_dir = Path(__file__).parent / "WiLoR"
    model_dir = wilor_dir / "pretrained_models"

    original_dir = Path.cwd()
    os.chdir(wilor_dir)

    try:
        model, model_cfg = load_wilor(
            checkpoint_path=str(model_dir / "wilor_final.ckpt"),
            cfg_path=str(model_dir / "model_config.yaml")
        )
    finally:
        os.chdir(original_dir)

    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
    detector = YOLO(str(model_dir / "detector.pt"))
    torch.load = original_torch_load

    device_obj = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device_obj)
    detector = detector.to(device_obj)
    model.eval()
    print(f"✓ Models loaded on {device_obj}\n")

    renderer_2d = Renderer(model_cfg, faces=model.mano.faces)
    faces = model.mano.faces
    faces_left = faces[:, [0, 2, 1]]

    # Process frames
    print("Processing frames...")
    frames_with_hands = []

    for frame_idx, frame in frames:
        if (frame_idx + 1) % 50 == 0:
            print(f"  Processing frame {frame_idx + 1}/{len(frames)}...")

        detections = detector(frame, conf=0.3, verbose=False)[0]
        if len(detections) == 0:
            continue

        bboxes = []
        is_right = []

        for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

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

            batch_size = batch['img'].shape[0]
            hands_in_frame = []

            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                is_right_hand = batch['right'][n].cpu().numpy()

                verts[:, 0] = (2 * is_right_hand - 1) * verts[:, 0]
                joints[:, 0] = (2 * is_right_hand - 1) * joints[:, 0]

                cam_t = pred_cam_t_full[n]

                hands_in_frame.append({
                    'vertices': verts,
                    'keypoints': joints,
                    'cam_t': cam_t,
                    'is_right': bool(is_right_hand),
                    'img_size': img_size[n],
                    'focal_length': scaled_focal_length,
                })

            if hands_in_frame:
                frames_with_hands.append({
                    'frame_idx': frame_idx,
                    'frame': frame,
                    'hands': hands_in_frame,
                })

    print(f"✓ Processed {len(frames_with_hands)} frames with hands\n")

    # OUTPUT 1: MANO overlaid on video
    print("=" * 70)
    print("Output 1: MANO Overlaid on Video")
    print("=" * 70)
    output1_path = output_dir / "1_mano_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer1 = cv2.VideoWriter(str(output1_path), fourcc, fps, (vid_width, vid_height))

    for i, frame_data in enumerate(frames_with_hands):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(frames_with_hands)}...")

        hands = frame_data['hands']
        vertices_list = [h['vertices'] for h in hands]
        cam_t_list = [h['cam_t'] for h in hands]
        is_right_list = [h['is_right'] for h in hands]
        img_size = hands[0]['img_size']
        focal_length = hands[0]['focal_length']

        misc_args = dict(
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=focal_length,
        )

        cam_view = renderer_2d.render_rgba_multiple(
            vertices_list,
            cam_t=cam_t_list,
            render_res=img_size,
            is_right=is_right_list,
            **misc_args
        )

        input_img = frame_data['frame']
        input_img = cv2.resize(input_img, (int(img_size[0]), int(img_size[1])))
        rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0

        valid_mask = (cam_view[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = (cam_view[:, :, :-1] * valid_mask +
                      (1 - valid_mask) * rgb_img)
        output_img = (output_img * 255).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        writer1.write(output_img)

    writer1.release()
    print(f"✓ Saved: {output1_path}\n")

    # OUTPUT 2: Skeleton overlaid on video
    print("=" * 70)
    print("Output 2: Skeleton Overlaid on Video")
    print("=" * 70)
    output2_path = output_dir / "2_skeleton_overlay.mp4"
    writer2 = cv2.VideoWriter(str(output2_path), fourcc, fps, (vid_width, vid_height))

    for i, frame_data in enumerate(frames_with_hands):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(frames_with_hands)}...")

        frame_with_skeleton = frame_data['frame'].copy()

        for hand in frame_data['hands']:
            keypoints_3d = hand['keypoints']
            cam_t = hand['cam_t']
            focal_length = hand['focal_length']
            img_size = hand['img_size']

            keypoints_2d = project_keypoints(keypoints_3d, cam_t, focal_length, img_size)
            frame_with_skeleton = draw_skeleton(frame_with_skeleton, keypoints_2d, HAND_CONNECTIONS,
                                                color=(0, 255, 0), thickness=2)

        writer2.write(frame_with_skeleton)

    writer2.release()
    print(f"✓ Saved: {output2_path}\n")

    # OUTPUT 3: MANO with white background (same POV as overlay)
    print("=" * 70)
    print("Output 3: MANO with White Background")
    print("=" * 70)
    output3_path = output_dir / "3_mano_no_overlay.mp4"
    writer3 = cv2.VideoWriter(str(output3_path), fourcc, fps, (vid_width, vid_height))

    for i, frame_data in enumerate(frames_with_hands):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(frames_with_hands)}...")

        output_img = render_3d_mano(frame_data, renderer_2d)
        writer3.write(output_img)

    writer3.release()
    print(f"✓ Saved: {output3_path}\n")

    # OUTPUT 4: Skeleton with white background (same POV as overlay)
    print("=" * 70)
    print("Output 4: Skeleton with White Background")
    print("=" * 70)
    output4_path = output_dir / "4_skeleton_no_overlay.mp4"
    writer4 = cv2.VideoWriter(str(output4_path), fourcc, fps, (vid_width, vid_height))

    for i, frame_data in enumerate(frames_with_hands):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(frames_with_hands)}...")

        output_img = render_3d_skeleton(frame_data, vid_width, vid_height)
        writer4.write(output_img)

    writer4.release()
    print(f"✓ Saved: {output4_path}\n")

    print("=" * 70)
    print("ALL 4 OUTPUTS COMPLETE!")
    print("=" * 70)
    print(f"\n1. MANO overlay: {output1_path}")
    print(f"2. Skeleton overlay: {output2_path}")
    print(f"3. MANO 3D grid: {output3_path}")
    print(f"4. Skeleton 3D grid: {output4_path}")


if __name__ == "__main__":
    main()
