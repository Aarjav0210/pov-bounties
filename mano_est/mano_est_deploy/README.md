# WiLoR Hand Estimation Environment

End-to-end 3D hand localization and reconstruction using WiLoR (CVPR 2025).

## Setup

The environment is already set up. If you need to reinstall:

```bash
pixi install
pixi run setup
```

## Models and Data

Located in `WiLoR/`:
- **MANO models** (`mano_data/`):
  - MANO_RIGHT.pkl (3.7M)
  - MANO_LEFT.pkl (3.7M)
- **Pretrained models** (`pretrained_models/`):
  - detector.pt (52M) - Hand detector
  - wilor_final.ckpt (2.4G) - WiLoR reconstruction model

## Usage

### Video Frame Processing

Process video frames and overlay hand meshes and skeletons:

```bash
# Activate environment
source activate.sh

# Process entire video with skeleton visualization
.pixi/envs/default/bin/python process_video_frames.py \
    -i path/to/video.mp4 \
    -o output_directory \
    --all-frames \
    --show-skeleton

# Process 20 random frames without skeleton
.pixi/envs/default/bin/python process_video_frames.py \
    -i path/to/video.mp4 \
    -o output_directory \
    -n 20
```

Options:
- `-i, --input`: Input video path (required)
- `-o, --output`: Output directory (default: hand_frames_output)
- `--all-frames`: Process all frames (default: sample random frames)
- `-n, --num-frames`: Number of random frames when not using --all-frames (default: 10)
- `--show-skeleton`: Draw 21-keypoint hand skeleton in green
- `--save-original`: Also save original frames alongside overlays
- `--rescale-factor`: Padding factor for hand bounding box (default: 2.0)
- `--seed`: Random seed for frame sampling (default: 42)

Output:
- `frame_XXXXX.jpg` - Frame with 3D hand mesh overlay (and skeleton if enabled)
- Frames without hands are skipped automatically

### 3D Trajectory Visualization

Extract 3D hand data and create trajectory visualizations:

```bash
# Extract data and create 3D trajectory video
.pixi/envs/default/bin/python extract_hand_3d.py \
    -i path/to/video.mp4 \
    -o output_directory \
    --output-video trajectory_3d.mp4

# Extract data only (for later visualization)
.pixi/envs/default/bin/python extract_hand_3d.py \
    -i path/to/video.mp4 \
    -o output_directory \
    --extract-only

# Visualize existing data
.pixi/envs/default/bin/python extract_hand_3d.py \
    --visualize-only output_directory/hand_data_3d.pkl \
    --output-video new_trajectory.mp4
```

Options:
- `-i, --input`: Input video path (required unless using --visualize-only)
- `-o, --output`: Output directory (default: hand_3d_output)
- `--output-video`: Output video file path (if not specified, shows interactive plot)
- `--extract-only`: Only extract 3D data, don't create visualization
- `--visualize-only DATA_FILE`: Create visualization from existing data file
- `--rescale-factor`: Padding factor for hand bounding box (default: 2.0)

Output:
- `hand_data_3d.pkl` - Pickled 3D data (vertices, keypoints, camera transforms)
- `trajectory_3d.mp4` - 3D visualization video showing hand mesh and skeleton trajectories

Features:
- MANO mesh vertices (778 points) shown as scatter plot
- 21 hand keypoints with skeleton connections in green
- Left hand in red, right hand in blue
- Rotating 3D view synchronized with video framerate

### Batch Processing Images

Process images from a folder:

```bash
# Activate environment
source activate.sh

# Run demo on images
.pixi/envs/default/bin/python WiLoR/demo.py \
    --img_folder demo_img \
    --out_folder demo_out \
    --save_mesh
```

Options:
- `--img_folder`: Input folder with images
- `--out_folder`: Output directory
- `--save_mesh`: Save 3D mesh files (.obj)

### Interactive Demo

Launch web-based demo:

```bash
source activate.sh
.pixi/envs/default/bin/python WiLoR/gradio_demo.py
```

This opens a local web interface for uploading images and visualizing 3D hand reconstructions.

## Output

The system generates:
- 3D hand meshes in MANO parametric form
- Visualizations with detected hands
- .obj mesh files (if `--save_mesh` enabled)

## Requirements

- 2x RTX 3090 GPUs (24GB each)
- CUDA 11.7
- PyTorch 2.0+

## Citation

WiLoR is a CVPR 2025 paper from Imperial College London and Shanghai Jiao Tong University. See the [original repository](https://github.com/rolpotamias/WiLoR) for citation details.
