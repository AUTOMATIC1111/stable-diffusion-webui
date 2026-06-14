# Stable Diffusion fundamentals

## What diffusion does
Stable Diffusion starts from random noise in **latent space** (a compressed representation of an image) and iteratively denoises it guided by your text prompt until a recognizable image emerges.

## Core terms

| Term | Meaning |
|------|---------|
| **Checkpoint** | Full trained model weights (UNet + text encoder + VAE bundled or partial) |
| **Latent space** | Lower-dimensional grid the UNet denoises (not pixel space) |
| **Text encoder** | Converts prompt tokens to conditioning vectors (CLIP for SD1.5) |
| **VAE** | Encodes/decodes between pixels and latents |
| **Positive prompt** | What you want |
| **Negative prompt** | What to suppress |
| **Seed** | RNG seed — same seed + settings → reproducible image |
| **Sampler** | Algorithm for each denoise step (euler, dpmpp_2m, etc.) |
| **Scheduler** | Noise schedule paired with sampler |
| **Steps** | Number of denoise iterations (more ≠ always better) |
| **CFG / guidance** | How strongly prompt constrains image (typical 5–8) |
| **Width / height** | Output resolution (affects VRAM quadratically) |
| **Denoising strength** | img2img: how much to change input (0=none, 1=full) |
| **LoRA** | Small add-on weights for style/subject |
| **ControlNet** | Structural guidance from edges, depth, pose, etc. |

## VRAM vs unified memory
- **NVIDIA VRAM**: dedicated GPU memory; OOM stops generation
- **Apple unified memory**: CPU and GPU share pool; pressure causes swapping

Continue to [02-first-image.md](02-first-image.md).
