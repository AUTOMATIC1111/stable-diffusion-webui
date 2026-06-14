# Upscaling and restoration

## Latent upscale
Upscale in latent space before final VAE decode — efficient.

## Pixel upscale
ESRGAN/UltraSharp in `models/comfyui/upscale_models/`.

## When not to upscale
Heavily artifacted base — upscaler amplifies flaws.

Workflow: `workflows/comfyui/upscale.json`.
