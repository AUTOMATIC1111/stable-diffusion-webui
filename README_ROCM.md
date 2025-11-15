# ROCm Setup Guide for Stable Diffusion WebUI

This guide helps you set up and optimize Stable Diffusion WebUI for AMD GPUs using ROCm 6.2.

## Quick Start

### 1. Copy the Optimized Launch Configuration

```bash
cp webui-user-rocm62.sh webui-user.sh
```

### 2. Launch the WebUI

```bash
./webui.sh
```

The launch script will automatically:
- Install PyTorch with ROCm 6.2 support
- Configure optimal VRAM settings for 16GB GPUs
- Set up memory management to prevent fragmentation

### 3. Configure WebUI Settings

After the WebUI starts, navigate to **Settings → Optimizations** and configure:

- **Cross attention optimization:** `Doggettx` (default)
- **Enable quantization in K samplers:** ✓ Enabled
- **Token merging ratio:** `0.5`

See the [VRAM Optimization Guide](ROCM_VRAM_OPTIMIZATION.md) for detailed configuration instructions.

---

## System Requirements

### Supported AMD GPUs

- **RX 6000 Series** (Navi 2): RX 6700 XT, 6800, 6800 XT, 6900 XT
- **RX 7000 Series** (Navi 3): RX 7600, 7700 XT, 7800 XT, 7900 XT, 7900 XTX
- **RX 5000 Series** (Navi 1): RX 5700 XT (with limitations)

### Recommended VRAM

- **Minimum:** 8GB VRAM
- **Recommended:** 16GB VRAM
- **Optimal:** 24GB VRAM

### Software Requirements

- **ROCm:** 6.2 or newer
- **Python:** 3.10 or 3.11
- **Linux:** Ubuntu 22.04, Fedora 38+, or Arch Linux

---

## Installation

### Option 1: Automatic Setup (Recommended)

The `webui.sh` script automatically detects AMD GPUs and installs ROCm 6.2 support.

```bash
# Clone the repository (if not already done)
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui

# Copy the optimized configuration
cp webui-user-rocm62.sh webui-user.sh

# Launch (will install dependencies automatically)
./webui.sh
```

### Option 2: Manual Setup

If you need manual control over the installation:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with ROCm 6.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install Stable Diffusion WebUI requirements
pip install -r requirements_versions.txt

# Set environment variables
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Launch with optimized flags
python launch.py --skip-torch-cuda-test --medvram --opt-split-attention --no-half-vae
```

---

## Configuration Files

### `webui-user-rocm62.sh`

Pre-configured launch script with optimal settings for AMD GPUs with 16GB VRAM.

**Key settings:**
- PyTorch with ROCm 6.2
- Memory fragmentation prevention
- VRAM-optimized command-line arguments

### `ROCM_VRAM_OPTIMIZATION.md`

Comprehensive guide covering:
- WebUI settings optimization
- Generation settings for different VRAM amounts
- ControlNet optimization
- Workflows for best quality
- Troubleshooting common issues

---

## Command-Line Arguments Explained

The optimized configuration uses these flags:

```bash
--skip-torch-cuda-test    # Skip CUDA test (we're using ROCm/HIP)
--medvram                 # Optimized for 8-16GB VRAM
--opt-split-attention     # Reduces VRAM usage during attention
--no-half-vae             # Prevents VAE errors with full precision
```

### For Different VRAM Amounts

**16GB VRAM (Recommended):**
```bash
--skip-torch-cuda-test --medvram --opt-split-attention --no-half-vae
```

**8GB VRAM:**
```bash
--skip-torch-cuda-test --lowvram --opt-split-attention --no-half-vae
```

**6GB VRAM or less:**
```bash
--skip-torch-cuda-test --lowvram --opt-split-attention --no-half-vae --opt-channelslast
```

---

## Recommended Generation Settings

### For 16GB VRAM

**Safe Mode (Fast, No Errors):**
- Resolution: 512x512
- Hires fix: OFF
- Batch size: 1
- VRAM usage: ~4-6GB

**Quality Mode (Best Results):**
- Resolution: 512x512
- Hires fix: ON (1.5x upscale)
- Hires steps: 10
- VRAM usage: ~8-12GB

**With ControlNet:**
- Resolution: 512x512
- Hires fix: OFF
- ControlNet units: 1-2 maximum
- Low VRAM mode: ON
- VRAM usage: ~6-10GB

See the [VRAM Optimization Guide](ROCM_VRAM_OPTIMIZATION.md) for detailed workflows and settings.

---

## Verification

### Check PyTorch ROCm Installation

```bash
source venv/bin/activate
python -c "import torch; print('ROCm available:', torch.cuda.is_available()); print('ROCm version:', torch.version.hip)"
```

**Expected output:**
```
ROCm available: True
ROCm version: 6.2.x
```

### Monitor VRAM Usage

```bash
watch -n 1 rocm-smi
```

Or check current usage:
```bash
rocm-smi --showmeminfo vram
```

---

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. **Reduce resolution:** 768x768 → 512x512
2. **Disable Hires fix** or reduce upscale ratio
3. **Use more aggressive flags:**
   ```bash
   export COMMANDLINE_ARGS="--skip-torch-cuda-test --lowvram --opt-split-attention --no-half-vae"
   ```

### Black Images or Artifacts

Ensure `--no-half-vae` is in your command-line arguments.

### Slow Generation

- Use `--medvram` instead of `--lowvram` for 16GB VRAM
- Reduce sampling steps to 20
- Try faster samplers: DPM++ 2M, Euler a

### Model Loading Errors

Verify PyTorch installation:
```bash
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

If it returns `False`, reinstall PyTorch:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

For more troubleshooting, see the [VRAM Optimization Guide](ROCM_VRAM_OPTIMIZATION.md#troubleshooting).

---

## Performance Tips

1. **Two-Phase Workflow:**
   - Generate at 512x512 without Hires fix (fast)
   - Upscale separately using img2img or Extras tab (best quality)

2. **ControlNet Best Practices:**
   - Use only 1-2 units at a time
   - Enable Low VRAM mode
   - Disable Hires fix when using ControlNet

3. **Batch Processing:**
   - Use `Batch count` instead of `Batch size`
   - Keep resolution at 512x512 for batches

4. **Memory Management:**
   - Restart WebUI after 50-100 generations
   - Use "Unload SD checkpoint" when switching models

---

## Additional Resources

- **[VRAM Optimization Guide](ROCM_VRAM_OPTIMIZATION.md)** - Comprehensive optimization guide
- **[ROCm Documentation](https://rocm.docs.amd.com/)** - Official AMD ROCm docs
- **[SD WebUI Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)** - Official WebUI documentation
- **[AMD GPU Support](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs)** - WebUI AMD GPU guide

---

## Summary

✅ **Key Points:**
- Use ROCm 6.2 for best compatibility
- Enable `expandable_segments:True` to prevent memory fragmentation
- Use `--medvram` for 16GB VRAM
- Start with 512x512, upscale separately for quality
- Enable ControlNet Low VRAM mode

❌ **Avoid:**
- Batch size > 1 (use Batch count instead)
- Hires fix with 2x upscale on 16GB VRAM
- More than 2 ControlNet units simultaneously
- Direct generation at resolutions > 768x768

---

**For detailed configuration and workflows, see [ROCM_VRAM_OPTIMIZATION.md](ROCM_VRAM_OPTIMIZATION.md)**
