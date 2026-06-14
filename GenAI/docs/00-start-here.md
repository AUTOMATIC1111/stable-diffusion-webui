# Start here

Welcome to **GenAI** — a self-contained local stack for:

1. **ComfyUI** — generate and edit images with Stable Diffusion workflows  
2. **FaceFusion** — swap faces in images and videos (authorized use only)

## Prerequisites

### Windows (RTX 4090 target)
- Windows 10/11, 64-bit
- [Git for Windows](https://git-scm.com/download/win)
- Python 3.10–3.12 with **Add to PATH**
- Latest [NVIDIA driver](https://www.nvidia.com/drivers)
- [FFmpeg](https://ffmpeg.org/download.html) (for FaceFusion video)
- ~50 GB free disk (apps + models)

### macOS (Apple Silicon target)
- macOS 13+ recommended
- Xcode Command Line Tools / Homebrew
- Python 3.10–3.12, Git, FFmpeg (`brew install ffmpeg python@3.12 git`)

## Installation path

| Step | Windows | macOS |
|------|---------|-------|
| 1. Enter project | `cd GenAI` | `cd GenAI` |
| 2. Setup | `.\scripts\windows\setup-all.ps1` | `./scripts/macos/setup-all.sh` |
| 3. Verify | `.\scripts\windows\doctor.ps1` | `./scripts/macos/doctor.sh` |
| 4. Add a checkpoint | See [models/README.md](../models/README.md) | Same |
| 5. First image | [02-first-image.md](stable-diffusion/02-first-image.md) | Same |

## Learning paths

**Stable Diffusion (ComfyUI)**  
→ [01-fundamentals.md](stable-diffusion/01-fundamentals.md) → [02-first-image.md](stable-diffusion/02-first-image.md) → [11-comfyui-fundamentals.md](stable-diffusion/11-comfyui-fundamentals.md)

**FaceFusion**  
→ [01-overview.md](facefusion/01-overview.md) → [02-first-image-face-swap.md](facefusion/02-first-image-face-swap.md) → [03-video-face-swap.md](facefusion/03-video-face-swap.md)

## Important boundaries

- This project **does not** replace the parent AUTOMATIC1111 WebUI in the repo root.
- **Roop is not supported.** Use FaceFusion only.
- Default network binding is **localhost** — do not expose to LAN without understanding the risk.
- **Never commit** private faces, videos, or model weights to Git.

## Get help

- Diagnostics: `doctor.ps1` / `doctor.sh`
- Static tests: `smoke-test.ps1` / `smoke-test.sh`
- Troubleshooting: [stable-diffusion/15-troubleshooting.md](stable-diffusion/15-troubleshooting.md), [facefusion/09-troubleshooting.md](facefusion/09-troubleshooting.md)
