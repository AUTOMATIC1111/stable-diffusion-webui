# TODO

## Issues

Stuff to be fixed, in no particular order...

- SD-XL VAE `AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)`
- SD-XL Lora
- SD-XL Sketch/Inpaint
- Kandinsky 2.2 (2.1 is working)
- Misterious Extensions auto-enabling
- Misterious Extra network corruptions
- script_callbacks.on_model_loaded 

## Features

Stuff to be added, in no particular order...

- Update `Wiki`
- Create new `GitHub` hooks/actions for CI/CD  
- Import core repos
- Update `train.py` to use `interrogator`
- Update `train.py` to use `rembg`
- Create new **Lora** train UI
- Docker PR
- Port `p.all_hr_prompts`
- Image watermark using `image-watermark`
- Image phash and hdash using `imagehash`
- Model merge using `git-rebasin`
- Additional upscalers
- XYZ grid upscalers
- New image browser
- Update `gradio`
- Rename repo: **automatic** -> **sdnext**
- New icons
- Enable refiner workflow for `ldm` backend
- Improve `lyco` logging
- Cache models when switching backends
- Style editor
- Built-in motd-style notifications

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
