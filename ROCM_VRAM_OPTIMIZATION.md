# ROCm 6.2 VRAM Optimization Guide for AMD GPUs

This guide provides comprehensive instructions for optimizing Stable Diffusion WebUI on AMD GPUs with ROCm 6.2, specifically targeting systems with 8-16GB VRAM.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Launch Configuration](#launch-configuration)
3. [WebUI Settings Optimization](#webui-settings-optimization)
4. [Generation Settings](#generation-settings)
5. [ControlNet Optimization](#controlnet-optimization)
6. [Recommended Workflows](#recommended-workflows)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Setup Launch Script

Copy the optimized ROCm 6.2 configuration:

```bash
cp webui-user-rocm62.sh webui-user.sh
```

### 2. Launch WebUI

```bash
./webui.sh
```

### 3. Configure WebUI Settings

Navigate to **Settings → Optimizations** and apply the recommended settings (see below).

---

## Launch Configuration

### Environment Variables

The following environment variables are set in `webui-user-rocm62.sh`:

#### PyTorch with ROCm 6.2

```bash
export TORCH_COMMAND="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
```

**Purpose:** Installs PyTorch compiled with ROCm 6.2 support for AMD GPUs.

#### HIP Memory Allocation

```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

**Purpose:** Prevents memory fragmentation, which is critical for stable VRAM usage and avoiding OOM (Out of Memory) errors.

### Command Line Arguments

```bash
export COMMANDLINE_ARGS="--skip-torch-cuda-test --medvram --opt-split-attention --no-half-vae"
```

#### Flag Explanations

| Flag | Purpose | VRAM Impact |
|------|---------|-------------|
| `--skip-torch-cuda-test` | Skip CUDA test (using ROCm/HIP instead) | N/A |
| `--medvram` | Optimized for 8-16GB VRAM, moves models between GPU/CPU as needed | **High** - Critical for 16GB |
| `--opt-split-attention` | Reduces VRAM usage during attention computation | **Medium** - Saves 1-2GB |
| `--no-half-vae` | Uses full precision for VAE to prevent errors | **Low** - Prevents artifacts |

#### Alternative Configurations

**For GPUs with less than 8GB VRAM:**

```bash
export COMMANDLINE_ARGS="--skip-torch-cuda-test --lowvram --opt-split-attention --no-half-vae"
```

**For maximum compatibility (slower but most stable):**

```bash
export COMMANDLINE_ARGS="--skip-torch-cuda-test --lowvram --opt-split-attention --no-half-vae --opt-channelslast"
```

**With xformers (if installed):**

```bash
export COMMANDLINE_ARGS="--skip-torch-cuda-test --medvram --xformers --no-half-vae"
```

---

## WebUI Settings Optimization

Navigate to **Settings → Optimizations** in the WebUI and configure:

### Recommended Settings

| Setting | Value | Notes |
|---------|-------|-------|
| **Cross attention optimization** | `Doggettx` or `xformers` | Doggettx is default and works well |
| **Enable quantization in K samplers** | ✓ Enabled | Reduces VRAM usage |
| **Token merging ratio** | `0.5` | Merges similar tokens to save memory |
| **Pad prompt/negative prompt** | ✓ Enabled | Recommended for consistency |

### Optional Advanced Settings

| Setting | Value | Effect |
|---------|-------|--------|
| **Token merging ratio for hires** | `0.5` | Saves VRAM during hires fix |
| **Always discard next-to-last sigma** | ✓ Enabled | Minor VRAM savings |

---

## Generation Settings

### Safe Mode (No Errors, 16GB VRAM)

**Best for:** Testing prompts, finding good seeds, general use

```
Width: 512
Height: 512
Batch count: 1
Batch size: 1
Hires fix: ☐ Disabled
Sampling steps: 20-30
```

**VRAM Usage:** ~4-6GB

---

### Quality Mode (With Hires Fix, 16GB VRAM)

**Best for:** Final high-quality outputs

```
Width: 512
Height: 512
Batch count: 1
Batch size: 1
Hires fix: ✓ Enabled
  Upscale by: 1.5 (avoid 2.0 on 16GB)
  Hires steps: 10
  Denoising strength: 0.4
  Upscaler: Latent or R-ESRGAN 4x+
Sampling steps: 20-30
```

**VRAM Usage:** ~8-12GB

**⚠️ Warning:** Using `Upscale by: 2.0` may cause OOM errors on 16GB VRAM.

---

### Portrait Mode (512x768)

**Best for:** Character portraits

```
Width: 512
Height: 768
Batch count: 1
Batch size: 1
Hires fix: ☐ Disabled (enable only after finding good seed)
Sampling steps: 20-30
```

**VRAM Usage:** ~6-8GB

---

### Landscape Mode (768x512)

**Best for:** Scenery, backgrounds

```
Width: 768
Height: 512
Batch count: 1
Batch size: 1
Hires fix: ☐ Disabled (enable only after finding good seed)
Sampling steps: 20-30
```

**VRAM Usage:** ~6-8GB

---

## ControlNet Optimization

When using ControlNet extensions, additional VRAM optimizations are necessary.

### ControlNet Settings

Navigate to **Settings → ControlNet** and configure:

| Setting | Value | Purpose |
|---------|-------|---------|
| **Low VRAM mode** | ✓ Enabled | Critical for 16GB VRAM |
| **Pixel Perfect** | ☐ Disabled | Disable during testing to save VRAM |
| **Control Mode** | `Balanced` | Default, good balance |

### Recommended Usage

```
Width: 512
Height: 512
Hires fix: ☐ Disabled
Active ControlNet units: 1-2 maximum
Batch size: 1
```

**⚠️ Warning:** Using 3+ ControlNet units simultaneously may cause OOM errors.

### ControlNet with Hires Fix

**Not recommended for 16GB VRAM.** If necessary:

```
Width: 512
Height: 512
Hires fix: ✓ Enabled
  Upscale by: 1.25 (minimum upscale)
  Hires steps: 5 (minimum steps)
Active ControlNet units: 1 maximum
Low VRAM mode: ✓ Enabled
```

---

## Recommended Workflows

### Workflow 1: Prompt Development (Fast)

**Goal:** Find the perfect prompt and seed quickly

1. **Settings:**
   - Size: 512x512
   - Hires fix: OFF
   - Steps: 20
   - Batch count: 4-8 (generate multiple images)

2. **Process:**
   - Experiment with different prompts
   - Test various seeds
   - Adjust CFG scale and sampling method

3. **VRAM Usage:** ~4-6GB per image

---

### Workflow 2: High-Quality Output (Two-Phase)

**Goal:** Maximum quality without VRAM errors

#### Phase 1: Generation

```
Size: 512x512
Hires fix: OFF
Steps: 30-40
Sampler: DPM++ 2M Karras or Euler a
CFG Scale: 7-8
```

**Find your perfect image** with the right prompt, seed, and composition.

#### Phase 2: Upscaling

**Option A: Using img2img**

1. Send image to img2img
2. Settings:
   - Resize to: 1024x1024 or 768x1152
   - Denoising: 0.3-0.5
   - Steps: 20-30
   - Sampler: Same as generation

**Option B: Using Extras Tab**

1. Send to Extras
2. Upscaler: R-ESRGAN 4x+ or 4x-UltraSharp
3. Scale: 2x or 4x
4. Optional: GFPGAN or CodeFormer for face restoration

**VRAM Usage:** Phase 1: ~4-6GB, Phase 2: ~6-10GB (depends on final resolution)

---

### Workflow 3: ControlNet Generation

**Goal:** Use ControlNet without VRAM errors

1. **Initial Setup:**
   - Size: 512x512
   - Hires fix: OFF
   - ControlNet units: 1-2 maximum
   - Low VRAM: ON

2. **Generate base image:**
   - Steps: 20-30
   - Find good composition

3. **Upscale separately:**
   - Use img2img without ControlNet
   - Or use Extras tab

**VRAM Usage:** ~6-10GB (depends on ControlNet type)

---

### Workflow 4: Batch Processing

**Goal:** Generate multiple images efficiently

**Small batches (recommended):**

```
Size: 512x512
Batch count: 4
Batch size: 1
Hires fix: OFF
```

**VRAM Usage:** ~4-6GB per image (sequential)

**⚠️ Avoid:**
- `Batch size > 1` (generates simultaneously, uses much more VRAM)
- Hires fix with batch processing

---

## Troubleshooting

### Issue: Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: HIP out of memory
```

**Solutions:**

1. **Reduce image resolution:**
   - 768x768 → 512x512
   - 512x768 → 512x512

2. **Disable Hires fix or reduce upscale:**
   - Turn OFF Hires fix
   - Or change `Upscale by: 2.0` → `1.5` or `1.25`

3. **Use more aggressive VRAM flags:**
   ```bash
   export COMMANDLINE_ARGS="--skip-torch-cuda-test --lowvram --opt-split-attention --no-half-vae"
   ```

4. **Reduce ControlNet units:**
   - Use only 1 ControlNet unit
   - Ensure Low VRAM mode is enabled

5. **Close other applications:**
   - Close browsers, games, or other GPU-intensive apps
   - Check `rocm-smi` to see VRAM usage

---

### Issue: Slow Generation Speed

**Symptoms:**
- Images take very long to generate
- System feels sluggish

**Solutions:**

1. **Check if you're using the right flags:**
   - Use `--medvram` not `--lowvram` for 16GB VRAM
   - `--lowvram` is slower but uses less VRAM

2. **Reduce sampling steps:**
   - Try 20 steps instead of 40-50
   - Use faster samplers: DPM++ 2M, Euler a

3. **Disable Token Merging:**
   - Settings → Optimizations → Token merging ratio: 0
   - Token merging saves VRAM but may slow down generation

4. **Check PyTorch installation:**
   ```bash
   python -c "import torch; print(torch.version.hip)"
   ```
   Should output ROCm version (e.g., `6.2.x`)

---

### Issue: Black Images or Artifacts

**Symptoms:**
- Generated images are black
- Strange artifacts or noise

**Solutions:**

1. **Enable `--no-half-vae`:**
   ```bash
   export COMMANDLINE_ARGS="--skip-torch-cuda-test --medvram --opt-split-attention --no-half-vae"
   ```

2. **Try different VAE:**
   - Settings → Stable Diffusion → SD VAE
   - Select `None` or try a different VAE

3. **Check cross attention optimization:**
   - Settings → Optimizations → Cross attention optimization
   - Try `Doggettx`, `sub-quadratic`, or `none`

---

### Issue: Model Loading Errors

**Symptoms:**
```
Error loading model
Couldn't load model
```

**Solutions:**

1. **Verify PyTorch ROCm installation:**
   ```bash
   source venv/bin/activate
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"
   ```
   Should output: `True` and ROCm version

2. **Reinstall PyTorch with ROCm 6.2:**
   ```bash
   source venv/bin/activate
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```

3. **Check model file integrity:**
   - Re-download the model
   - Verify SHA256 hash if available

---

### Issue: Memory Fragmentation

**Symptoms:**
- VRAM usage increases over time
- OOM errors after multiple generations

**Solutions:**

1. **Ensure expandable segments is enabled:**
   ```bash
   export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
   ```

2. **Restart the WebUI periodically:**
   - After 50-100 generations, restart the WebUI

3. **Use the "Unload SD checkpoint" button:**
   - Settings → Actions → Unload SD checkpoint to free VRAM
   - Useful when switching between models

---

### Checking VRAM Usage

**Monitor VRAM in real-time:**

```bash
watch -n 1 rocm-smi
```

**Check current VRAM usage:**

```bash
rocm-smi --showmeminfo vram
```

---

## Performance Benchmarks

Approximate generation times on AMD RX 6800/6900 XT (16GB VRAM):

| Configuration | Resolution | Hires Fix | Steps | Time |
|---------------|------------|-----------|-------|------|
| Safe Mode | 512x512 | No | 20 | ~8-12s |
| Safe Mode | 512x512 | No | 30 | ~12-18s |
| Quality Mode | 512x512 → 768x768 | Yes (1.5x) | 20+10 | ~20-30s |
| Quality Mode | 512x512 → 1024x1024 | Yes (2x) | 20+10 | ~35-50s |
| Portrait | 512x768 | No | 20 | ~12-16s |
| ControlNet | 512x512 | No | 20 | ~15-25s |

*Times may vary based on model, sampler, and prompt complexity.*

---

## Additional Resources

- **ROCm Documentation:** https://rocm.docs.amd.com/
- **Stable Diffusion WebUI Wiki:** https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki
- **AMD GPU Support:** https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs

---

## Summary of Key Points

✅ **DO:**
- Use `--medvram` for 16GB VRAM
- Enable `expandable_segments:True` to prevent fragmentation
- Start with 512x512 resolution
- Use Hires fix with `1.5x` upscale maximum
- Enable ControlNet Low VRAM mode
- Generate at low resolution, upscale separately for best quality

❌ **DON'T:**
- Use `Batch size > 1` (use `Batch count` instead)
- Use `Upscale by: 2.0` with Hires fix on 16GB VRAM
- Enable 3+ ControlNet units simultaneously
- Generate at 1024x1024 or higher directly
- Forget to set `--no-half-vae` (prevents VAE errors)

---

**Last Updated:** 2025-11-15
**ROCm Version:** 6.2
**Target VRAM:** 8-16GB
