# Stable Diffusion WebUI - Development Workflow Guide

## Overview

This guide explains how to integrate Stable Diffusion image generation into your development workflow for your portfolio projects.

## Quick Reference

### Start WebUI (One-time)
```bash
cd stable-diffusion-webui
python3.10 launch.py --xformers --api &
# Wait ~30 seconds for startup
```

### CLI Usage
```bash
# Basic generation
python3.10 automate_sd.py --prompt "your prompt" --output image.png

# With custom settings
python3.10 automate_sd.py --prompt "landscape" --steps 30 --width 1024 --height 576 --output hero.png

# Image-to-image
python3.10 automate_sd.py --prompt "oil painting" --img2img input.png --denoise 0.6 --output output.png
```

### Python API Usage
```python
from automate_sd import StableDiffusionAPI

api = StableDiffusionAPI()
api.wait_for_api()

# Generate image
images = api.txt2img(prompt="your prompt", steps=25, width=512, height=512)
api.save_image(images[0], "output.png")
```

---

## Development Workflow Integration

### 1. Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Development Workflow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Start WebUI     →  python3.10 launch.py --xformers --api   │
│                           (runs in background)                   │
│                                                                  │
│  2. Generate Assets →  Use CLI or Python API                    │
│                                                                  │
│  3. Use in Project  →  Copy images to your portfolio           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Starting the WebUI

**Option A: Manual Start**
```bash
cd /home/tbaltzakis/new-portfolio/figma-cloud-portfolio/stable-diffusion-webui
python3.10 launch.py --xformers --api
```

**Option B: Background (recommended for dev)**
```bash
cd stable-diffusion-webui
nohup python3.10 launch.py --xformers --api > /tmp/sd.log 2>&1 &
echo "PID: $!"  # Save this PID to stop later
```

**Option C: With auto-start script**
```bash
# Create a start script
echo '#!/bin/bash
cd stable-diffusion-webui
nohup python3.10 launch.py --xformers --api > sd.log 2>&1 &
echo "WebUI starting at http://127.0.0.1:7860"' > start_sd.sh
chmod +x start_sd.sh
./start_sd.sh
```

**Stop the WebUI:**
```bash
pkill -f "launch.py"
# Or use the PID you saved
kill <PID>
```

### 3. Generating Images for Your Portfolio

#### A. Using the Portfolio Examples Script

```bash
# Run predefined portfolio generation
python3.10 examples/portfolio_examples.py

# Output locations:
# - outputs/portfolio/     (hero images)
# - outputs/thumbnails/    (project thumbnails)
# - outputs/placeholders/  (dev placeholders)
# - outputs/batch/        (custom batch)
```

#### B. Using the CLI Directly

**Hero Images (1024x576 - 16:9):**
```bash
python3.10 automate_sd.py \
  --prompt "modern tech dashboard" \
  --steps 30 \
  --width 1024 \
  --height 576 \
  --output outputs/hero1.png
```

**Thumbnails (512x512):**
```bash
python3.10 automate_sd.py \
  --prompt "app icon mobile design" \
  --steps 25 \
  --width 512 \
  --height 512 \
  --output outputs/thumb1.png
```

**Social Media (1080x1080):**
```bash
python3.10 automate_sd.py \
  --prompt "social media post design" \
  --steps 35 \
  --width 1080 \
  --height 1080 \
  --output outputs/social1.png
```

#### C. Using Python API in Your Projects

Create a script `generate_assets.py` in your project:

```python
#!/usr/bin/env python3
"""Generate portfolio assets for your project"""
import sys
import os

# Add Stable Diffusion WebUI to path
sys.path.insert(0, '/home/tbaltzakis/new-portfolio/figma-cloud-portfolio/stable-diffusion-webui')

from automate_sd import StableDiffusionAPI

def main():
    api = StableDiffusionAPI()
    if not api.wait_for_api():
        print("Error: WebUI not running")
        sys.exit(1)
    
    # Define your prompts
    assets = [
        {"prompt": "hero image for tech portfolio", "file": "hero.png", "size": (1024, 576)},
        {"prompt": "project thumbnail app design", "file": "thumb1.png", "size": (512, 512)},
        {"prompt": "project thumbnail web design", "file": "thumb2.png", "size": (512, 512)},
    ]
    
    for asset in assets:
        print(f"Generating: {asset['file']}")
        images = api.txt2img(
            prompt=asset["prompt"],
            steps=25,
            width=asset["size"][0],
            height=asset["size"][1]
        )
        api.save_image(images[0], asset["file"])
        print(f"Saved: {asset['file']}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python3.10 generate_assets.py
```

---

## Common Tasks

### Generate Multiple Variations
```bash
for i in 1 2 3 4 5; do
  python3.10 automate_sd.py \
    --prompt "modern minimalist logo" \
    --seed $i \
    --output "logo_v$i.png"
done
```

### Use Specific Sampler
```bash
python3.10 automate_sd.py \
  --prompt "landscape" \
  --sampler "DPM++ 2M" \
  --output image.png
```

### Image-to-Image (Style Transfer)
```bash
python3.10 automate_sd.py \
  --prompt "oil painting style" \
  --img2img existing_image.png \
  --denoise 0.5 \
  --output styled.png
```

---

## Best Practices

### 1. Image Quality Settings

| Use Case | Steps | CFG | Size |
|----------|-------|-----|------|
| Quick Preview | 15 | 7.0 | 512x512 |
| Standard | 20-25 | 7.0 | 512x512 |
| High Quality | 30+ | 7.5-8.0 | 768+ |
| Hero Images | 30+ | 8.0 | 1024x576 |

### 2. Prompt Tips

- **Be specific:** "modern minimalist website hero" not "website"
- **Add style:** "photorealistic", "vector art", "3D render"
- **Use negative prompts:** `--negative "blurry, low quality"`

### 3. Automation Script Template

```python
#!/usr/bin/env python3
"""Your custom image generation script"""
import sys
sys.path.insert(0, '/home/tbaltzakis/new-portfolio/figma-cloud-portfolio/stable-diffusion-webui')

from automate_sd import StableDiffusionAPI
import os

# Your prompts
PROMTS = [
    ("project 1 hero", "assets/p1_hero.png", 1024, 576),
    ("project 1 thumb", "assets/p1_thumb.png", 512, 512),
    ("project 2 hero", "assets/p2_hero.png", 1024, 576),
]

def main():
    api = StableDiffusionAPI()
    api.wait_for_api()
    
    for prompt, path, w, h in PROMPTS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        images = api.txt2img(prompt=prompt, steps=25, width=w, height=h)
        api.save_image(images[0], path)
        print(f"✓ {path}")

if __name__ == "__main__":
    main()
```

---

## Troubleshooting

### WebUI won't start
```bash
# Check logs
tail -50 /tmp/sdwebui.log

# Restart
pkill -f "launch.py"
cd stable-diffusion-webui
python3.10 launch.py --api  # without xformers
```

### API not responding
```bash
# Verify WebUI is running
curl http://127.0.0.1:7860/sdapi/v1/samplers

# Check if port is in use
lsof -i :7860
```

### Out of memory
```bash
# Use smaller images
--width 512 --height 512

# Or reduce batch size in automate_sd.py
```

---

## File Structure

```
stable-diffusion-webui/
├── automate_sd.py              # Main automation script
├── examples/
│   └── portfolio_examples.py   # Portfolio generation examples
├── outputs/                    # Generated images
│   ├── portfolio/
│   ├── thumbnails/
│   └── placeholders/
├── modules/
│   ├── paths.py               # Module mocks (fixed)
│   └── sd_hijack_unet.py     # xformers fix
└── launch.py                   # WebUI launcher
```

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| Start WebUI | `python3.10 launch.py --xformers --api` |
| Generate image | `python3.10 automate_sd.py --prompt "..." --output x.png` |
| List samplers | `curl -s http://127.0.0.1:7860/sdapi/v1/samplers` |
| Stop WebUI | `pkill -f "launch.py"` |
| Check status | `curl -s http://127.0.0.1:7860/sdapi/v1/options` |

---

## Integration with Your Portfolio

1. Generate images using the tools above
2. Copy from `outputs/` to your portfolio's `public/` or `assets/` folder
3. Reference in your Next.js/React components:
   ```jsx
   <Image src="/assets/hero.png" width={1024} height={576} />
   ```
