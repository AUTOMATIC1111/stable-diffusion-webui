# TODO

## Issues

Stuff to be fixed, in no particular order...

N/A

## Features

Stuff to be added, in no particular order...

- Diffusers:
  - Add Lyco support
  - Add ControlNet
  - Fix DeepFloyd IF model
  - Add unCLIP model
  - Add Training support
  - Add long prompts
- Technical debt:
  - Port **A1111** stuff
  - Port `p.all_hr_prompts`
  - Import core repos to reduce dependencies
- Non-technical:
  - Update Wiki
  - Rename repo: **automatic** -> **sdnext**
  - [Localization](https://app.transifex.com/signup/open-source/)
- New Minor
  - Prompt padding for positive/negative
  - Add EN provider for VAEs
  - Built-in `motd`-style notifications
- New Major
  - Profile manager (for `config.json` and `ui-config.json`)
  - Multi-user support
  - Add [SAG](https://huggingface.co/docs/diffusers/v0.19.3/en/api/pipelines/self_attention_guidance), [SAG](https://github.com/ashen-sensored/sd_webui_SAG)
  - Image phash and hdash using `imagehash`
  - Model merge using `git-rebasin`
  - Enable refiner-style workflow for `ldm` backend
  - Add `sgm` backend
  - Cache models in VRAM
  - Train:
    - Use `interrogator`
    - Use `rembg`
    - Templates for SD-XL training
    - Lora train UI
- Redesign
  - New UI
  - New inpainting canvas controls (move from backend to purely frontend)
  - New image browser (move from backend to purely frontend)
  - Change workflows from static/legacy to steps-based

## Investigate

Stuff to be investigated...

## Merge PRs

Pick & merge PRs from main repo...

- up-to-date with: df004be
- current todo list: <https://github.com/AUTOMATIC1111/stable-diffusion-webui/compare/df004be...394ffa7>

## Integration

Tech that can be integrated as part of the core workflow...

- [Git-ReBasin]([https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion](https://github.com/vladmandic/automatic/issues/1176))
- [Null-text inversion](https://github.com/ouhenio/null-text-inversion-colab)
- [Custom diffusion](https://github.com/guaneec/custom-diffusion-webui), [Custom diffusion](https://www.cs.cmu.edu/~custom-diffusion/)
- [Dream artist](https://github.com/7eu7d7/DreamArtist-sd-webui-extension)
- [QuickEmbedding](https://github.com/ethansmith2000/QuickEmbedding)
- [DragGAN](https://github.com/XingangPan/DragGAN)
- [LamaCleaner](https://github.com/Sanster/lama-cleaner)
- `TensorRT`

## Random

- Bunch of stuff: <https://pharmapsychotic.com/tools.html>
