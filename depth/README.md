# Video Depth Anything Environment

Temporal consistent depth estimation using Video Depth Anything V2 (CVPR 2025 Highlight).

## Setup

The environment uses Pixi for dependency management. To set up:

```bash
# Install dependencies and download models
pixi install
pixi run setup
```

## Environment Configuration

This environment has been configured to resolve several compatibility issues:

### Fixed Issues:
1. **NumPy Version**: Locked to `1.26.4` (NumPy 2.x causes compatibility issues)
2. **PyTorch/TorchVision**: Specific versions `2.6.0` and `0.21.0` for CUDA 11.8 compatibility
3. **xFormers Disabled**: Force-disabled in all attention modules to avoid CUDA memory issues
4. **Dependencies**: All required packages (opencv, einops, imageio, decord, etc.) managed by pixi.toml

### Why xFormers is Disabled:
xFormers has compatibility issues with PyTorch versions and can cause CUDA out-of-memory errors. The code has been modified to use standard PyTorch attention instead, which works reliably with the `vits` (small) model.

## Models Available

Located in `Video-Depth-Anything/checkpoints/`:
- `depth_anything_v2_vits.pth` - Small model (28M params, 112M, ~7GB VRAM)
- `depth_anything_v2_vitb.pth` - Base model (113M params, ~12GB VRAM)
- `depth_anything_v2_vitl.pth` - Large model (381M params, 1.5G, ~23GB VRAM)

**Recommended for H100**: Use `vitl` (large) model for best quality depth estimation. With 80GB VRAM, the H100 can easily handle the ~23GB requirement.

**For GPUs with limited VRAM**: Use `vits` (small) model which requires only ~7GB VRAM.

## Usage

Process a video to generate depth maps:

```bash
# Activate environment
source activate.sh

# Run depth estimation (Large model - RECOMMENDED for H100)
.pixi/envs/default/bin/python Video-Depth-Anything/run.py \
    --input_video path/to/video.mp4 \
    --output_dir ./outputs \
    --encoder vitl

# Run depth estimation (Base model)
.pixi/envs/default/bin/python Video-Depth-Anything/run.py \
    --input_video path/to/video.mp4 \
    --output_dir ./outputs \
    --encoder vitb

# Run depth estimation (Small model - for limited VRAM)
.pixi/envs/default/bin/python Video-Depth-Anything/run.py \
    --input_video path/to/video.mp4 \
    --output_dir ./outputs \
    --encoder vits
```

### Options

- `--encoder`: Model size (`vits` for Small, `vitb` for Base, `vitl` for Large)
- `--input_video`: Path to input video file
- `--output_dir`: Directory to save depth outputs
- `--grayscale`: Save grayscale depth maps instead of colored

## Requirements

- NVIDIA GPU with CUDA support
- Minimum 8GB VRAM for `vits` model
- 12GB VRAM for `vitb` model
- 24GB+ VRAM for `vitl` model
- CUDA 12.x
- PyTorch 2.6.0 with CUDA 11.8 binaries

## Output

The script generates:
- Depth map video (colored or grayscale)
- Individual depth map frames
- Temporal consistency metrics

## Troubleshooting

### CUDA Out of Memory
- Use the `vits` model instead of `vitl`
- Reduce video resolution
- Process shorter video clips

### NumPy Compatibility Errors
- Ensure NumPy 1.26.4 is installed (not 2.x)
- Reinstall environment: `pixi install`

### xFormers Errors
- xFormers is intentionally disabled in this environment
- If you see xFormers-related errors, verify the force-disable flags in:
  - `video_depth_anything/dinov2_layers/attention.py`
  - `video_depth_anything/motion_module/attention.py`
  - `video_depth_anything/motion_module/motion_module.py`
