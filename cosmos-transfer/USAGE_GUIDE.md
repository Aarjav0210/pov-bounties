# Cosmos-Transfer 2.5: Egocentric Video Domain Randomization Guide

## Overview
You've successfully set up **Cosmos-Transfer2.5** for processing egocentric monocular videos into domain-randomized versions for robotic training. The system uses **edge-based control only** (no depth grounding required!).

## Your Setup
- **GPU**: NVIDIA H100 80GB
- **CUDA**: 12.1
- **Python**: 3.10 (virtual environment)
- **Model**: Cosmos-Transfer2.5-2B (edge control variant)
- **Installation Path**: `/home/paperspace/cosmos-transfer/cosmos-transfer2.5/`

## Quick Start

### 1. Activate Environment
```bash
cd /home/paperspace/cosmos-transfer
source cosmos-transfer2.5/venv/bin/activate
```

### 2. Prepare Your Video
Your input video requirements:
- **Format**: MP4, AVI, MOV
- **Resolution**: 480p-720p (will be processed to 720p)
- **Frame Rate**: 16-30 FPS
- **Duration**: 2-5 seconds (optimal for training data)
- **Content**: Egocentric views (first-person perspective)

### 3. Create a Config File
Create a JSON config for your video (e.g., `my_video_config.json`):

```json
{
    "name": "my_egocentric_video",
    "prompt": "A detailed description of what's in your video",
    "video_path": "/absolute/path/to/your/video.mp4",
    "guidance": 3.0,
    "edge": {
        "control_weight": 1.0
    }
}
```

**Prompt Tips:**
- Describe the scene, objects, lighting, and environment
- Be specific about materials, colors, and spatial relationships
- Example: "A first-person view of hands manipulating objects on a kitchen counter with natural daylight"

### 4. Run Inference
```bash
cd /home/paperspace/cosmos-transfer/cosmos-transfer2.5
source venv/bin/activate
python examples/inference.py \
  -i /path/to/your/config.json \
  -o /path/to/output/directory
```

### 5. Monitor Progress
- First run: Model downloads (~5GB), takes 2-3 minutes
- Inference: ~12-15 minutes per 4-second video on H100
- Check GPU usage: `nvidia-smi`

## Key Parameters for Domain Randomization

### Guidance Scale (`guidance`)
- **Range**: 1.0 - 7.0
- **Low (1.0-2.0)**: More variation, less faithful to prompt
- **Medium (2.5-4.0)**: Balanced (recommended for domain randomization)
- **High (5.0-7.0)**: Strict adherence to prompt

### Edge Weight (`control_weight` in `edge`)
- **Range**: 0.0 - 1.5
- **Low (0.3-0.7)**: More freedom, different layouts
- **Medium (0.8-1.2)**: Balanced structure preservation (recommended)
- **High (1.3-1.5)**: Strict edge preservation

## Creating Training Data Variations

To generate multiple domain-randomized versions of the same video:

### Method 1: Vary the Prompt
Create multiple configs with different environmental descriptions:
```json
// Version 1: Bright laboratory
{"prompt": "A brightly lit modern laboratory with white surfaces", ...}

// Version 2: Dim workshop
{"prompt": "A dimly lit industrial workshop with metal surfaces", ...}

// Version 3: Outdoor setting
{"prompt": "An outdoor workspace with natural lighting and wooden surfaces", ...}
```

### Method 2: Vary Parameters
```json
// High variation
{"guidance": 2.0, "edge": {"control_weight": 0.5}}

// Balanced
{"guidance": 3.0, "edge": {"control_weight": 1.0}}

// Conservative
{"guidance": 4.0, "edge": {"control_weight": 1.3}}
```

## Batch Processing Multiple Videos

Create a JSONL file (one JSON object per line):
```bash
cat > batch_configs.jsonl << 'EOF'
{"name": "video1", "prompt": "Description 1", "video_path": "/path/to/video1.mp4", "guidance": 3.0, "edge": {"control_weight": 1.0}}
{"name": "video2", "prompt": "Description 2", "video_path": "/path/to/video2.mp4", "guidance": 3.0, "edge": {"control_weight": 1.0}}
EOF

# Run batch
python examples/inference.py -i batch_configs.jsonl -o outputs/batch
```

## Multi-GPU Inference (Optional)
If you have multiple GPUs:
```bash
torchrun --nproc_per_node=2 -m examples.inference \
  -i batch_configs.jsonl \
  -o outputs/batch
```

## Output Structure
```
outputs/
└── your_output_dir/
    ├── my_egocentric_video/
    │   ├── video.mp4          # Generated domain-randomized video
    │   ├── metadata.json      # Generation parameters
    │   └── controls/
    │       └── edge.mp4       # Edge map visualization
    └── logs/
```

## Troubleshooting

### Out of Memory
- Reduce video resolution to 480p
- Process shorter clips (2-3 seconds)
- Use single GPU mode

### Slow Generation
- Expected: ~12-15 min per video on H100
- Check GPU utilization: `nvidia-smi`
- Ensure no other processes are using GPU

### Edge Detection Issues
- Increase edge contrast in source video
- Adjust `control_weight` (try 0.7-1.3)
- Check input video quality

## Best Practices for Robotic Training Data

1. **Diversity**: Generate 5-10 variations per source video
2. **Consistency**: Keep edge_weight ≥ 0.8 to maintain spatial relationships
3. **Realism**: Use guidance 2.5-4.0 for photorealistic results
4. **Coverage**: Vary lighting, materials, and environments in prompts

## Example Workflow

```bash
# 1. Place your egocentric videos in a folder
mkdir -p /home/paperspace/my_videos
cp your_video.mp4 /home/paperspace/my_videos/

# 2. Create configs for 5 variations
for i in {1..5}; do
  cat > config_v${i}.json << EOF
{
    "name": "robot_task_v${i}",
    "prompt": "Variation ${i}: [your custom description]",
    "video_path": "/home/paperspace/my_videos/your_video.mp4",
    "guidance": $(echo "2.5 + $i * 0.3" | bc),
    "edge": {"control_weight": 1.0}
}
EOF
done

# 3. Run all variations
cd /home/paperspace/cosmos-transfer/cosmos-transfer2.5
source venv/bin/activate
for config in /home/paperspace/cosmos-transfer/config_v*.json; do
    python examples/inference.py -i $config -o /home/paperspace/outputs
done
```

## Need Help?
- Documentation: `/home/paperspace/cosmos-transfer/cosmos-transfer2.5/docs/`
- Examples: `/home/paperspace/cosmos-transfer/cosmos-transfer2.5/assets/robot_example/`
- GitHub: https://github.com/nvidia-cosmos/cosmos-transfer2.5
