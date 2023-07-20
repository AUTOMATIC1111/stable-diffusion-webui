# TODO

## Issues

Stuff to be fixed, in no particular order...

- Misterious Extra network corruptions
- Unet type mismatch/autocast issues on some platforms

## Features

Stuff to be added, in no particular order...

- Diffusers:
  - Add SD-XL Lora
  - Add SD-XL Sketch/Inpaint
  - Add VAE direct load from safetensors
  - Fix Kandinsky 2.2 model
  - Fix DeepFloyd IF model
  - Redo Prompt parser
  - Add Explicit VAE step
  - Add Save image before refiner (depends on explicit VAE)
  - Add unCLIP model
- Technical debt:
  - Port **A1111** stuff
  - Port `p.all_hr_prompts`
  - Import core repos to reduce dependencies
  - Update `gradio`
- Non-technical:
  - Create additional themes
  - Update Wiki
  - Get more high-quality upscalers
  - Rename repo: **automatic** -> **sdnext**
  - [Localization](https://app.transifex.com/signup/open-source/)
- New Minor
  - Prompt padding for positive/negative
  - XYZ grid upscalers
  - Built-in `motd`-style notifications
  - Docker PR
- New Major
  - Style editor (use json format instead of csv)
  - Profile manager (for config.json and ui-config.json)
  - Multi-user support
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
  - New extra networks (move from backend to purely frontend)
  - Change workflows from static/legacy to steps-based

## Investigate

Stuff to be investigated...

## Merge PRs

Pick & merge PRs from main repo...

- up-to-date with: df004be
- current todo list: <https://github.com/AUTOMATIC1111/stable-diffusion-webui/compare/df004be...394ffa7>

## Integration

Tech that can be integrated as part of the core workflow...

- [Merge without distortion](https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion)
- [Weighted merges](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui/tree/master)
- [Null-text inversion](https://github.com/ouhenio/null-text-inversion-colab)
- [Custom diffusion](https://github.com/guaneec/custom-diffusion-webui), [Custom diffusion](https://www.cs.cmu.edu/~custom-diffusion/)
- [Dream artist](https://github.com/7eu7d7/DreamArtist-sd-webui-extension)
- [QuickEmbedding](https://github.com/ethansmith2000/QuickEmbedding)
- [DataComp CLiP](https://github.com/mlfoundations/open_clip/blob/main/docs/datacomp_models.md)
- [ClipSeg](https://github.com/timojl/clipseg)
- [DragGAN](https://github.com/XingangPan/DragGAN)
- [LamaCleaner]([Title](https://github.com/Sanster/lama-cleaner))
- `TensorRT`

## Random

- Bunch of stuff: <https://pharmapsychotic.com/tools.html>
- <https://towardsdatascience.com/mastering-memoization-in-python-dcdd8b435189>
