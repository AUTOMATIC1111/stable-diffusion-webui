# Stable Diffusion WebUI - Automation Guide

## Overview

This directory contains an automated Python script (`automate_sd.py`) that provides programmatic access to the Stable Diffusion WebUI API for generating images without needing the graphical interface.

## Prerequisites

1. **WebUI must be running** with the API enabled:
   ```bash
   cd stable-diffusion-webui
   python3.10 launch.py --xformers --api
   ```

2. **Required Python packages**:
   ```bash
   pip install requests
   ```

## Quick Start

### Starting the WebUI

```bash
cd /home/tbaltzakis/new-portfolio/figma-cloud-portfolio/stable-diffusion-webui
python3.10 launch.py --xformers --api
```

The WebUI will be available at: **http://127.0.0.1:7860**

### Basic Image Generation

```bash
python3.10 automate_sd.py --prompt "a beautiful landscape" --output landscape.png
```

### Advanced Options

```bash
# Higher quality (more steps)
python3.10 automate_sd.py --prompt "portrait" --steps 50 --output portrait.png

# Larger images
python3.10 automate_sd.py --prompt "cinematic scene" --width 768 --height 768 --output scene.png

# Using a specific seed for reproducibility
python3.10 automate_sd.py --prompt "cat" --seed 42 --output cat.png

# Image-to-image (transform an existing image)
python3.10 automate_sd.py --prompt "oil painting style" --img2img input.png --denoise 0.6 --output output.png
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prompt` | Text description of the image to generate | (required) |
| `--negative` | Things to avoid in the image | "" |
| `--output` | Output filename | "output.png" |
| `--steps` | Number of denoising steps (higher = better quality) | 20 |
| `--cfg` | CFG scale (7 is good default) | 7.0 |
| `--width` | Image width (must be multiple of 8) | 512 |
| `--height` | Image height (must be multiple of 8) | 512 |
| `--seed` | Random seed (-1 for random) | -1 |
| `--sampler` | Sampling method | "Euler a" |
| `--url` | WebUI URL | http://127.0.0.1:7860 |
| `--img2img` | Input image for image-to-image generation | (none) |
| `--denoise` | Denoising strength for img2img (0-1) | 0.75 |

## Python API Usage

You can also use the automation programmatically in your own Python scripts:

```python
from automate_sd import StableDiffusionAPI

# Connect to WebUI
api = StableDiffusionAPI("http://127.0.0.1:7860")

# Wait for API to be ready
api.wait_for_api()

# Generate an image
images = api.txt2img(
    prompt="a sunset over mountains",
    steps=20,
    cfg_scale=7.0,
    width=512,
    height=512
)

# Save the image
api.save_image(images[0], "sunset.png")

# Image-to-image
images = api.img2img(
    prompt="make it more abstract",
    image_path="input.png",
    denoising_strength=0.7
)
```

## Available Samplers

Common samplers (specified with `--sampler`):
- `Euler a` - Fast, good quality (default)
- `Euler` - Classic Euler method
- `DPM++ 2M` - High quality, recommended
- `DPM++ 2M Karras` - DPM++ with Karras noise schedule
- `DDIM` - Good for image-to-image
- `PLMS` - Legacy sampler

## Troubleshooting

### WebUI not starting

If you see `ModuleNotFoundError: No module named 'taming'`, the fixes have already been applied. Make sure you're in the correct directory and running the command properly:

```bash
cd /home/tbaltzakis/new-portfolio/figma-cloud-portfolio/stable-diffusion-webui
python3.10 launch.py --xformers --api
```

### API connection error

Ensure the WebUI is running and accessible at the correct URL. The default is `http://127.0.0.1:7860`.

### xformers error

If you encounter xformers compatibility issues, try running without xformers:

```bash
python3.10 launch.py --api
```

## Files Modified

The following files were modified/created to fix startup issues and enable automation:

- `modules/paths.py` - Added taming module mocks
- `modules/sd_hijack_unet.py` - Fixed xformers compatibility  
- `repositories/stable-diffusion-stability-ai/ldm/data/util.py` - Added AddMiDaS stub
- `repositories/stable-diffusion-stability-ai/ldm/models/ddpm.py` - Added LatentDepth2ImageDiffusion stub
- `automate_sd.py` - Automation script (NEW)
- `AUTOMATION_README.md` - This documentation
