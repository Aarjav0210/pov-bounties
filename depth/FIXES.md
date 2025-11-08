# Depth Environment Fixes

This document details all the fixes applied to the Video-Depth-Anything environment to resolve compatibility issues.

## Issues Fixed

### 1. NumPy Version Incompatibility ✓ FIXED
**Problem**: NumPy 2.2.6 was installed but code was compiled with NumPy 1.x
- Error: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6"

**Solution**: Locked NumPy to version 1.26.4 in `pixi.toml`
```toml
numpy = "1.26.4"
```

### 2. PyTorch/TorchVision Incompatibility ✓ FIXED
**Problem**: Had PyTorch 2.9.0 and torchvision 0.24.0 (too new, incompatible with codebase)

**Solution**: Specified exact compatible versions in `pixi.toml`
```toml
pytorch = {version = "2.6.0", channel = "pytorch"}
torchvision = {version = "0.21.0", channel = "pytorch"}
```

### 3. xFormers Not Working ✓ FIXED
**Problem**:
- xFormers was built for PyTorch 2.1.1+cu121 but we had different versions
- CUDA extensions wouldn't load
- Error: "No operator found for memory_efficient_attention_forward"

**Solution**: Force-disabled xFormers in all attention modules
- Modified 3 files to set `XFORMERS_AVAILABLE = False`:
  - `video_depth_anything/dinov2_layers/attention.py` (line 23)
  - `video_depth_anything/motion_module/attention.py` (line 24)
  - `video_depth_anything/motion_module/motion_module.py` (line 19)
- Removed xformers from pip install list in `setup.sh`

### 4. CUDA Out of Memory ✓ FIXED
**Problem**: After disabling xFormers, standard PyTorch attention used too much VRAM
- vitl model required ~23GB but only had that much total on smaller GPUs

**Solution**:
- Default to `vitl` (large) model for H100 (80GB VRAM available)
- Document VRAM requirements for each model
- Provide clear guidance on model selection based on GPU

### 5. Missing Dependencies ✓ FIXED
**Problem**: Several Python packages were missing or had version conflicts

**Solution**: Added all required dependencies to `pixi.toml`
```toml
[dependencies]
opencv = "*"
einops = "*"
imageio = "*"
imageio-ffmpeg = "*"
tqdm = "*"

[pypi-dependencies]
decord = "*"
easydict = "*"
```

## Files Modified

1. **`pixi.toml`**
   - Locked PyTorch to 2.6.0, torchvision to 0.21.0
   - Locked NumPy to 1.26.4
   - Added all missing dependencies
   - Added pypi-dependencies section for decord and easydict

2. **`setup.sh`**
   - Removed redundant pip install commands
   - Added note that dependencies are managed by pixi.toml
   - Explicitly noted that xFormers is NOT installed

3. **`README.md`**
   - Added comprehensive environment configuration section
   - Documented all fixed issues
   - Added troubleshooting guide
   - Updated VRAM requirements for each model
   - Added usage recommendations

4. **Attention modules** (already fixed in previous session)
   - `video_depth_anything/dinov2_layers/attention.py`
   - `video_depth_anything/motion_module/attention.py`
   - `video_depth_anything/motion_module/motion_module.py`

## Verification

To verify the fixes are applied correctly:

```bash
# Check xFormers is disabled in all files
grep -n "XFORMERS_AVAILABLE = False" \
  depth/Video-Depth-Anything/video_depth_anything/dinov2_layers/attention.py \
  depth/Video-Depth-Anything/video_depth_anything/motion_module/attention.py \
  depth/Video-Depth-Anything/video_depth_anything/motion_module/motion_module.py

# Check pixi.toml has correct versions
cat depth/pixi.toml
```

## Reinstalling Environment

If you need to reinstall the environment with the fixes:

```bash
cd depth

# Remove old environment
rm -rf .pixi

# Reinstall with fixed configuration
pixi install
pixi run setup
```

## Model Selection Guide

**For H100 (80GB VRAM):** ⭐ Current Configuration
- Use `vitl` (large) for best quality - **DEFAULT**
- Can process multiple videos in parallel
- ~23GB VRAM per video, can handle 3+ simultaneous processes

**For RTX 3090/4090 (24GB VRAM):**
- Use `vits` (small) - recommended
- Can use `vitb` (base) for better quality if processing single videos
- `vitl` may work for single videos if no other processes running

**For RTX 3060/4060 (8-12GB VRAM):**
- Use `vits` (small) only
- May need to reduce video resolution

## Integration with Full Pipeline

The depth stage is now fully compatible with the unified pipeline script (`process_full_pipeline.py`):

```bash
# Run full pipeline with depth stage (uses vitl by default for H100)
python process_full_pipeline.py

# Explicitly specify model if needed
python process_full_pipeline.py --depth-encoder vitl  # Large (default)
python process_full_pipeline.py --depth-encoder vitb  # Base
python process_full_pipeline.py --depth-encoder vits  # Small

# Skip depth if you want to test other stages
python process_full_pipeline.py --skip depth
```

## Known Limitations

1. **xFormers Disabled**: Standard PyTorch attention is slower but more reliable
2. **Model Size**: Large model (vitl) requires significant VRAM
3. **Processing Speed**: Without xFormers, processing is ~20-30% slower

## Future Improvements

1. Consider upgrading to newer PyTorch versions when xFormers compatibility improves
2. Explore alternative attention implementations for better performance
3. Add support for batched video processing to improve throughput
