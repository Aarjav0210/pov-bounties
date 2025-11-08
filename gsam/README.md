# Grounded Segment Anything - Video Segmentation

Text-prompted video segmentation using Grounding DINO + Segment Anything Model (SAM).

## Overview

This pipeline segments objects in videos using natural language text prompts. It combines:
- **Grounding DINO**: Text-based object detection (finds objects matching text descriptions)
- **Segment Anything Model (SAM)**: Precise pixel-level segmentation masks

## Hardware Requirements

- **GPU**: Works well on 2x RTX 3090 (24GB each)
- **VRAM**: ~5GB for model, additional memory scales with video resolution
- **Memory Management**: Frame-by-frame processing to minimize VRAM usage

## Setup

```bash
# Navigate to gsam directory
cd /home/jay/Documents/Brown/Research/Cosmos/gsam

# Install pixi environment
pixi install

# Run setup (clones repos, installs deps, downloads models)
pixi run setup
```

Setup will download:
- Segment Anything ViT-H checkpoint (~2.4GB)
- Grounding DINO SwinT checkpoint (~700MB)

## Usage

### Basic Video Segmentation

```bash
pixi run segment -i video.mp4 -p "hand . object . table"
```

### Command Line Options

```
-i, --input          Input video path (required)
-p, --prompt         Text prompt for objects to segment (required)
                     Separate multiple objects with " . " (e.g. "hand . cup . table")
-o, --output         Output directory (default: outputs)
--box-threshold      Box confidence threshold (default: 0.35)
--text-threshold     Text confidence threshold (default: 0.25)
--no-masks           Skip saving individual mask images
--device             GPU device (default: cuda:0, can use cuda:1)
```

### Example: Segment Hands and Objects

```bash
pixi run segment \
  -i robot_demo.mp4 \
  -p "hand . gripper . object" \
  -o outputs/robot_demo \
  --box-threshold 0.4
```

### Example: Use Second GPU

```bash
pixi run segment \
  -i video.mp4 \
  -p "person . chair . desk" \
  --device cuda:1
```

## Output Structure

```
outputs/
├── video_segmented.mp4       # Annotated video with masks and boxes
├── detections.json            # Frame-by-frame detection metadata
└── masks/                     # Individual mask images (if --no-masks not set)
    ├── frame_0000/
    │   ├── mask_00.png
    │   └── mask_01.png
    ├── frame_0001/
    │   └── ...
    └── ...
```

### Detection Metadata Format

`detections.json` contains:
```json
{
  "video": "path/to/input.mp4",
  "text_prompt": "hand . object",
  "total_frames": 120,
  "detections": [
    {
      "frame": 0,
      "num_detections": 2,
      "boxes": [[x1, y1, x2, y2], ...]
    },
    ...
  ]
}
```

## Text Prompt Guide

- **Multiple objects**: Separate with " . " (space-dot-space)
  - Good: `"hand . cup . table"`
  - Bad: `"hand, cup, table"` or `"hand.cup.table"`

- **Be specific**: More specific prompts work better
  - Good: `"human hand . coffee mug . wooden table"`
  - Okay: `"hand . mug . table"`

- **Object parts**: Can specify parts of objects
  - `"left hand . right hand . fingertips"`

## Performance Notes

- **Processing Speed**: ~2-5 FPS depending on resolution and number of detections
- **Memory Usage**: GPU cache cleared every 30 frames to prevent OOM
- **Resolution**: Higher resolution videos use more VRAM but give better masks

## Model Details

- **SAM**: ViT-H (huge) variant - most accurate, requires ~2.4GB
- **Grounding DINO**: SwinT-OGC - balanced speed/accuracy, requires ~700MB
- **Total Model Size**: ~3.1GB on disk, ~5GB VRAM when loaded

## Troubleshooting

### Out of Memory
- Use `--device cuda:1` to use second GPU
- Reduce video resolution before processing
- Use `--no-masks` to skip saving individual masks

### Poor Detection Quality
- Adjust `--box-threshold` (higher = fewer but more confident detections)
- Adjust `--text-threshold` (higher = stricter text matching)
- Refine text prompt to be more specific

### No Detections
- Check text prompt format (use " . " separator)
- Lower `--box-threshold` and `--text-threshold`
- Verify objects are visible and clear in video frames

## Directory Structure

```
gsam/
├── pixi.toml              # Environment configuration
├── setup.sh               # Setup script
├── activate.sh            # Environment activation
├── segment_video.py       # Main segmentation script
├── README.md             # This file
├── weights/              # Model checkpoints (created by setup)
├── segment-anything/     # SAM repository (created by setup)
├── GroundingDINO/        # Grounding DINO repository (created by setup)
└── Grounded-Segment-Anything/  # Reference repo (created by setup)
```

## References

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
