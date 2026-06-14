# Models, checkpoints, and VAEs

## Families
- **SD 1.5**: 512 native, vast LoRA ecosystem
- **SDXL**: 1024 native, two-encoder pipeline
- **Flux / SD3**: newer architectures — verify ComfyUI node support

## Compatibility
Do not load SDXL checkpoint in SD1.5 workflow nodes.

## VAE
Some checkpoints include baked VAE; external VAE fixes color/face issues. Place in `models/comfyui/vae/`.

## Comparison method
Lock seed, steps, sampler; change only checkpoint for fair test.

See [../models/README.md](../models/README.md).
