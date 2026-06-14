# Windows performance (RTX 4090)

24 GB VRAM supports most SD1.5/SDXL single-image workflows at default settings.

## Typical VRAM
| Workflow | Approx VRAM |
|----------|-------------|
| SD1.5 512 txt2img | 4–6 GB |
| SDXL 1024 txt2img | 8–12 GB |
| + ControlNet | +2–4 GB |

## Bottlenecks
1. First-load checkpoint read from disk
2. Attention at high resolution
3. VAE decode when upscaling

## Tips
- Lock seed for comparisons ([06-samplers](../stable-diffusion/06-samplers-schedulers-steps-and-cfg.md))
- Watch `nvidia-smi` during first SDXL run
- ComfyUI cancel button stops queue; kill terminal only if hung

See also [stable-diffusion/13-performance-and-memory.md](../stable-diffusion/13-performance-and-memory.md).
