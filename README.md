# POV Bounties - MANO Hand Estimation Pipeline

This repository contains a pipeline for extracting 3D hand poses and MANO models from videos using [WiLoR (CVPR 2025)](https://github.com/rolpotamias/WiLoR).

## Overview

The pipeline processes videos and generates 4 outputs:
1. **MANO overlay** - 3D hand meshes overlaid on the original video
2. **Skeleton overlay** - Hand skeleton keypoints overlaid on the original video
3. **MANO (no overlay)** - 3D hand meshes with white background (same POV as overlay)
4. **Skeleton (no overlay)** - Hand skeleton with white background (same POV as overlay)

## Prerequisites

### Required Model Files

You need to download the following files from the [WiLoR repository](https://github.com/rolpotamias/WiLoR) and [MANO website](https://mano.is.tue.mpg.de):

#### 1. WiLoR Pretrained Models
Download these files and place them in `mano_est/mano_est_deploy/WiLoR/pretrained_models/`:

```bash
cd mano_est/mano_est_deploy/WiLoR
mkdir -p pretrained_models

# Download detector model (~52 MB)
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/

# Download WiLoR checkpoint (~2.4 GB)
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```

**Expected files:**
- `mano_est/mano_est_deploy/WiLoR/pretrained_models/detector.pt`
- `mano_est/mano_est_deploy/WiLoR/pretrained_models/wilor_final.ckpt`
- `mano_est/mano_est_deploy/WiLoR/pretrained_models/model_config.yaml` (already in repo)
- `mano_est/mano_est_deploy/WiLoR/pretrained_models/dataset_config.yaml` (already in repo)

#### 2. MANO Model Files
MANO models must be downloaded from the [official MANO website](https://mano.is.tue.mpg.de):

1. Create an account at https://mano.is.tue.mpg.de by clicking "Sign Up"
2. Download the MANO models (`mano_v*_*.zip`)
3. Unzip the downloaded file
4. Place the following files in `mano_est/mano_est_deploy/WiLoR/mano_data/`:
   - `MANO_RIGHT.pkl` (~3.7 MB)
   - `MANO_LEFT.pkl` (~3.7 MB)

**Expected files:**
- `mano_est/mano_est_deploy/WiLoR/mano_data/MANO_RIGHT.pkl`
- `mano_est/mano_est_deploy/WiLoR/mano_data/MANO_LEFT.pkl`
- `mano_est/mano_est_deploy/WiLoR/mano_data/mano_mean_params.npz` (already in repo)

**Important:** MANO models fall under the [MANO license](https://mano.is.tue.mpg.de/license.html). By using this repository, you must comply with their license terms.

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Aarjav0210/pov-bounties.git
cd pov-bounties
```

### 2. Set up the environment
```bash
cd mano_est/mano_est_deploy
source activate.sh
```

The `activate.sh` script will set up the Pixi environment with all required dependencies.

### 3. Download model files
Follow the instructions in the [Required Model Files](#required-model-files) section above.

## Usage

### Single Video Processing

Process a single video with the main script:

```bash
cd mano_est/mano_est_deploy
source activate.sh
python generate_all_4_outputs.py -i path/to/video.mp4 -o output_directory
```

**Arguments:**
- `-i, --input`: Path to input video file (required)
- `-o, --output`: Output directory (default: `final_outputs`)

**Output:**
The script will create 4 MP4 files in the output directory:
- `1_mano_overlay.mp4` - MANO meshes overlaid on video
- `2_skeleton_overlay.mp4` - Skeleton keypoints overlaid on video
- `3_mano_no_overlay.mp4` - MANO meshes with white background
- `4_skeleton_no_overlay.mp4` - Skeleton keypoints with white background

### Batch Processing

To process multiple videos at once:

1. Place all your videos in the `inputs/` folder:
```bash
cd /path/to/pov-bounties
mkdir -p inputs
cp *.mp4 inputs/
```

2. Run the batch processor:
```bash
python process_batch.py
```

**Output Structure:**
```
outputs/
  video1/
    1_mano_overlay.mp4
    2_skeleton_overlay.mp4
    3_mano_no_overlay.mp4
    4_skeleton_no_overlay.mp4
  video2/
    1_mano_overlay.mp4
    2_skeleton_overlay.mp4
    3_mano_no_overlay.mp4
    4_skeleton_no_overlay.mp4
```

## Project Structure

```
pov-bounties/
├── README.md                           # This file
├── process_batch.py                    # Batch processing script
├── inputs/                             # Input videos directory
├── outputs/                            # Batch output directory
└── mano_est/
    └── mano_est_deploy/
        ├── activate.sh                 # Environment activation
        ├── generate_all_4_outputs.py   # Main processing script
        └── WiLoR/                      # WiLoR submodule
            ├── pretrained_models/      # Download pretrained models here
            │   ├── detector.pt         # [DOWNLOAD REQUIRED]
            │   ├── wilor_final.ckpt    # [DOWNLOAD REQUIRED]
            │   ├── model_config.yaml
            │   └── dataset_config.yaml
            └── mano_data/              # Download MANO models here
                ├── MANO_RIGHT.pkl      # [DOWNLOAD REQUIRED]
                ├── MANO_LEFT.pkl       # [DOWNLOAD REQUIRED]
                └── mano_mean_params.npz
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~10 GB disk space for models
- Dependencies managed via Pixi (automatically installed with `activate.sh`)

## Technical Details

### Hand Representation
- **MANO Model**: Parametric 3D hand model with 778 vertices and 21 keypoints per hand
- **Skeleton**: 21 keypoints connected in hand topology (thumb, index, middle, ring, pinky)

### Coordinate Systems
- Camera intrinsics are preserved between overlay and non-overlay outputs
- Same focal length and camera center used for consistent perspective
- 180° rotation around X-axis transforms between coordinate systems

### Processing Pipeline
1. Video frames are extracted
2. Hands are detected using YOLOv8-based detector
3. WiLoR model estimates 3D hand pose and MANO parameters
4. Outputs are rendered with overlay or white background

## Troubleshooting

### "ERROR: Repository not found" when pushing
Make sure the git remote is set correctly:
```bash
git remote set-url origin git@github.com:Aarjav0210/pov-bounties.git
```

### Missing model files
Verify all required files exist:
```bash
ls -lh mano_est/mano_est_deploy/WiLoR/pretrained_models/
ls -lh mano_est/mano_est_deploy/WiLoR/mano_data/
```

### CUDA out of memory
Reduce the batch size in `generate_all_4_outputs.py` (line 404):
```python
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, ...)  # Reduce from 16
```

## License

This repository depends on:
- [WiLoR](https://github.com/rolpotamias/WiLoR) - CC-BY-NC-ND License
- [MANO Model](https://mano.is.tue.mpg.de/license.html) - MANO License
- [Ultralytics](https://github.com/ultralytics/ultralytics) - AGPL-3.0 License

By using this repository, you must comply with the terms of all external licenses.

## Citation

If you use this pipeline, please cite the WiLoR paper:

```bibtex
@misc{potamias2024wilor,
    title={WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild},
    author={Rolandos Alexandros Potamias and Jinglei Zhang and Jiankang Deng and Stefanos Zafeiriou},
    year={2024},
    eprint={2409.12259},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgements

This pipeline is built on top of:
- [WiLoR](https://github.com/rolpotamias/WiLoR) by Rolandos Alexandros Potamias et al.
- [MANO](https://mano.is.tue.mpg.de) hand model
- [Ultralytics](https://github.com/ultralytics/ultralytics) for hand detection
