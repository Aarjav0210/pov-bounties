# Cosmos-Transfer Integration Plan

## Overview
Set up Cosmos-Transfer2.5 to process videos from `inputs/` directory, generate photorealistic augmented views, and save outputs to `outputs/` with filename `{videoname}_aug.mp4`.

## Current Understanding

### Cosmos-Transfer2.5 Architecture
- **Purpose**: Multi-controlnet model for generating photorealistic videos from structured control inputs
- **Control Types**: edge, depth, segmentation (seg), blur (vis)
- **Key Feature**: Can compute edge and blur controls on-the-fly from input video
- **Model Size**: Cosmos-Transfer2-2B (requires 65.4 GB VRAM)

### Directory Structure
```
cosmos-transfer/
├── cosmos-transfer2.5/
│   ├── assets/                    # Example inputs and prompts
│   │   ├── robot_example/
│   │   └── car_example/
│   ├── cosmos_transfer2/
│   │   ├── inference.py           # Main inference module
│   │   ├── config.py              # Configuration classes
│   │   └── ...
│   ├── examples/
│   │   └── inference.py           # CLI entry point
│   └── docs/
│       ├── inference.md           # Inference guide
│       └── setup.md              # Setup instructions
└── activate.sh                    # Environment activation
```

### Input JSON Format
Cosmos-Transfer2.5 expects JSON files with this structure:

```json
{
  "name": "sample_name",
  "prompt": "Detailed text description of desired output video",
  "negative_prompt": "What to avoid in generation (optional)",
  "seed": 0,
  "guidance": 3,
  "edge": {
    "control_weight": 1.0
    // control_path is optional - will be computed on-the-fly if not provided
  }
}
```

**Key Fields:**
- `name`: Unique identifier for the sample
- `prompt`: Text description of the desired photorealistic output
- `negative_prompt`: Optional negative guidance (default: cartoonish, pixelated, etc.)
- `edge`: Edge control configuration (computed automatically from input video if not specified)
- `depth`, `seg`, `vis`: Additional optional control inputs

### Current Inference Command
```bash
# Single GPU
python examples/inference.py -i <json_file> -o <output_dir>

# Multi-GPU (8 GPUs)
torchrun --nproc_per_node=8 --master_port=12341 -m examples.inference -i <json_file> -o <output_dir>
```

## Implementation Plan

### Phase 1: Core Infrastructure

#### 1.1 Create JSON Prompt Generator
**File**: `cosmos-transfer/generate_prompts.py`

**Purpose**: Generate JSON configuration files for each video in `inputs/`

**Functionality**:
- Scan `inputs/` directory for .mp4 files
- For each video:
  - Check if corresponding `.json` file exists
  - If not, generate default JSON with:
    - `name`: video filename (without extension)
    - `prompt`: Placeholder or AI-generated description
    - `negative_prompt`: Standard negative prompt
    - `edge`: Auto-compute configuration
- Save JSON to `inputs/{videoname}.json`

**Prompt Generation Strategy**:
1. **Option A (Simple)**: Use placeholder prompts that users can edit
2. **Option B (Advanced)**: Use vision-language model to describe video content
3. **Option C (Hybrid)**: Provide template prompts based on video category

**Example Output** (`inputs/gcsample.json`):
```json
{
  "name": "gcsample",
  "prompt": "A high-quality photorealistic video showing hand manipulation and gestures. The scene features natural lighting, realistic textures, and smooth motion. The hands appear lifelike with proper skin tones and detailed movements.",
  "negative_prompt": "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all.",
  "seed": 0,
  "guidance": 3,
  "edge": {
    "control_weight": 1.0
  }
}
```

#### 1.2 Create Batch Processing Script
**File**: `cosmos-transfer/process_cosmos_batch.py`

**Purpose**: Process all videos in `inputs/` through Cosmos-Transfer2.5

**Workflow**:
1. Activate cosmos-transfer environment
2. Scan `inputs/` for videos and corresponding JSON files
3. For each video+JSON pair:
   - Prepare inference configuration
   - Run `examples/inference.py` with appropriate parameters
   - Monitor output generation
4. Post-process outputs:
   - Rename generated video to `{videoname}_aug.mp4`
   - Move to `outputs/` directory
5. Generate summary report

**Key Features**:
- Validate JSON configurations before processing
- Handle multi-GPU setup if available
- Error handling and logging
- Progress tracking
- Resource monitoring (GPU memory)

#### 1.3 Modify Inference Script (if needed)
**File**: `cosmos-transfer/cosmos-transfer2.5/examples/inference.py`

**Potential Modifications**:
- Accept video path directly in JSON configuration
- Customize output naming scheme
- Add progress callbacks
- Integrate with batch processor

### Phase 2: Integration with Existing Pipeline

#### 2.1 Update Root Batch Processor
**File**: `/home/paperspace/pov-bounties/process_batch.py`

**Add Cosmos-Transfer Step**:
```python
# After MANO processing
1. Run MANO hand estimation → outputs/videoname/*.mp4
2. Run Cosmos-Transfer augmentation → outputs/videoname/*_aug.mp4
```

#### 2.2 Unified Input/Output Structure
```
pov-bounties/
├── inputs/                        # Input videos and prompts
│   ├── video1.mp4
│   ├── video1.json               # Cosmos-Transfer prompt
│   ├── video2.mp4
│   ├── video2.json
│   └── ...
├── outputs/                       # All outputs
│   ├── video1/
│   │   ├── 1_mano_overlay.mp4          # MANO outputs
│   │   ├── 2_skeleton_overlay.mp4
│   │   ├── 3_mano_no_overlay.mp4
│   │   ├── 4_skeleton_no_overlay.mp4
│   │   └── video1_aug.mp4              # Cosmos-Transfer output
│   └── video2/
│       ├── ...
│       └── video2_aug.mp4
├── mano_est/                      # MANO pipeline
└── cosmos-transfer/               # Cosmos-Transfer pipeline
    ├── generate_prompts.py        # NEW: Generate JSON prompts
    ├── process_cosmos_batch.py    # NEW: Batch processor
    └── cosmos-transfer2.5/        # Existing codebase
```

### Phase 3: Prompt Engineering

#### 3.1 Default Prompt Template
Create high-quality default prompts for different video types:

**Hand Manipulation Videos**:
```
"A photorealistic video demonstration of precise hand movements and gestures.
The scene features natural indoor lighting with soft shadows, capturing realistic
skin tones and textures. The hands move smoothly and naturally, showcasing dexterity
and coordination. The background is clean and well-lit, with professional video
quality. Camera remains steady, focusing on the detailed hand motions."
```

**Driving/Outdoor Videos**:
```
"A high-quality driving scene captured in a modern urban environment. The video
features realistic road conditions, accurate vehicle dynamics, and natural outdoor
lighting. Buildings and infrastructure have photorealistic textures. The scene
includes proper shadows, reflections, and atmospheric perspective. Camera movement
is smooth and stable."
```

#### 3.2 Prompt Customization Options
- Allow users to edit JSON files before processing
- Provide prompt library for common scenarios
- Support custom negative prompts per video

### Phase 4: Optimization & Quality Control

#### 4.1 Performance Optimization
- **GPU Memory Management**: Monitor VRAM usage (requires 65.4 GB)
- **Batch Processing**: Process multiple videos sequentially
- **Checkpointing**: Save intermediate results to resume on failure

#### 4.2 Quality Validation
- Check output video quality (resolution, framerate)
- Validate generated videos aren't corrupted
- Compare input/output frame counts
- Log generation metrics (time, memory usage)

## Technical Requirements

### Hardware
- **GPU**: NVIDIA GPU with ≥65.4 GB VRAM (H100, B200, or multi-GPU setup)
- **Storage**: Sufficient space for input videos, outputs, and model checkpoints

### Software Dependencies
- Cosmos-Transfer2.5 environment (managed via activate.sh)
- PyTorch with CUDA support
- FFmpeg for video processing
- Python 3.10+

### Model Checkpoints
- Cosmos-Transfer2-2B checkpoint (auto-downloaded via setup)
- Edge detection model (optional, computed on-the-fly)

## Implementation Steps

### Step 1: Create Prompt Generator
```bash
cd cosmos-transfer
python generate_prompts.py --input-dir ../inputs --template hand_manipulation
```

### Step 2: Test Single Video
```bash
cd cosmos-transfer2.5
python examples/inference.py -i ../../inputs/gcsample.json -o ../../outputs/gcsample
```

### Step 3: Create Batch Processor
```bash
cd cosmos-transfer
python process_cosmos_batch.py --input-dir ../inputs --output-dir ../outputs
```

### Step 4: Integrate with Main Pipeline
```bash
cd /home/paperspace/pov-bounties
python process_batch.py  # Runs both MANO and Cosmos-Transfer
```

## Expected Outputs

For each input video, generate:
1. **MANO Outputs** (existing):
   - `1_mano_overlay.mp4`
   - `2_skeleton_overlay.mp4`
   - `3_mano_no_overlay.mp4`
   - `4_skeleton_no_overlay.mp4`

2. **Cosmos-Transfer Output** (new):
   - `{videoname}_aug.mp4` - Photorealistic augmented video

## Success Criteria

- ✅ Automatic JSON generation for all input videos
- ✅ Successful inference on test videos
- ✅ Proper output naming and organization
- ✅ Error handling and logging
- ✅ Reasonable generation time (5-15 minutes per video on H100)
- ✅ High-quality photorealistic outputs

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Insufficient GPU memory | Use multi-GPU setup or smaller batches |
| Slow generation times | Optimize inference parameters, use faster GPUs |
| Poor output quality | Improve prompts, adjust control weights |
| Missing model checkpoints | Verify setup, download required models |
| JSON format errors | Validate JSON before inference |

## Next Steps

1. **Immediate**: Create `generate_prompts.py` script
2. **Test**: Run single video through pipeline
3. **Scale**: Create batch processor
4. **Integrate**: Update main process_batch.py
5. **Document**: Update README with Cosmos-Transfer usage

## Questions to Resolve

1. What should default prompts look like for hand videos?
2. Should we use multi-GPU setup or single GPU?
3. Do we want to customize control weights per video?
4. Should we generate depth maps or rely on edge detection only?
5. How to handle videos without corresponding JSON files?

## Resources

- [Cosmos-Transfer GitHub](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [Inference Documentation](cosmos-transfer2.5/docs/inference.md)
- [Setup Guide](cosmos-transfer2.5/docs/setup.md)
