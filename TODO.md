# TODO

## Issues

Stuff to be fixed...


## Features

Stuff to be added...

- Update `README.md`
- Update `Wiki`
- Add `Gradio` theme maker
- Create new `GitHub` hooks/actions for CI/CD  
- Monitor file changes for misbehaving extensions
- Reload browser on server restart
- Remove origin wiki
- Import core repos
- Import rembg
- Improve core `Stability-AI` code: <https://github.com/vladmandic/automatic/discussions/795>
- Improve core `k-Diffusion` code

## Investigate

Stuff to be investigated...

- Gradio `app_kwargs`: <https://github.com/gradio-app/gradio/issues/4054>

## Merge PRs

Pick & merge PRs from main repo...

- Compare commits: <https://github.com/vladmandic/automatic/compare/e59ebe25..b6289d5>

## Models

StabilityAI is working on new stuff...

- SD XL
- SD ReImagined

## Integration

Tech that can be integrated as part of the core workflow...

- [Merge without distortion](https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion)
- [Weighted merges](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui/tree/master)
- [Null-text inversion](https://github.com/ouhenio/null-text-inversion-colab)
- [Custom diffusion](https://github.com/guaneec/custom-diffusion-webui), [Custom diffusion](https://www.cs.cmu.edu/~custom-diffusion/)
- [Dream artist](https://github.com/7eu7d7/DreamArtist-sd-webui-extension)
- [QuickEmbedding](https://github.com/ethansmith2000/QuickEmbedding)
- `TensorRT`

## Random

- Bunch of stuff: <https://pharmapsychotic.com/tools.html>

### Pending Code Updates

- tested with **torch 2.1** and **cuda 12.1**  
  (production remains on torch2.0.1+cuda11.8)  
- fully extend support of `--data-dir`  
  allows multiple installations to share pretty much everything, not just models  
- redo api authentication  
  now api authentication will use same user/pwd (if specified) for ui and strictly enforce it using httpbasicauth  
  new authentication is also fully supported in combination with ssl for both sync and async calls  
  if you want to use api programatically, see examples in `cli/sdapi.py`  
- add dark/light theme mode toggle  
- redo some `clip-skip` functionality  
- better matching for vae vs model  
- update to `xyz grid` to allow creation of large number of images without  
- update `gradio` (again)  
- more prompt parser optimizations  
- better error handling when importing image settings which are not compatible with current install  
  for example, when upscaler or sampler originally used is not available  
- fixes...amazing how many issues were introduced by porting new a1111 code without adding almost no new functionality  
