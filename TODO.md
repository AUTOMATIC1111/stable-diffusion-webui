# TODO

## Issues

Stuff to be fixed...

- Fix integration with <https://github.com/deforum-art/deforum-for-automatic1111-webui>

## Features

Stuff to be added...

- Add README headers
- Add Gradio base themes: <https://gradio.app/theming-guide/#using-the-theme-builder>
- Create new GitHub hooks/actions for CI/CD  
- Move Restart Server from WebUI to Launch and reload modules
- Redo Extensions tab: see <https://vladmandic.github.io/sd-extension-manager/pages/extensions.html>
- Stream-load models as option for slow storage
- Replace **PngInfo** / **EXIF** metadata handler
- Investigate integration with `Torch-DirectML`
- Investigate best practices for **Apple M1**
- Investigate best practices for **AMD GPUs**

## Investigate

Stuff to be investigated...

- Revisit `torch.compile`

## Merge PRs

Pick & merge PRs from main repo...

- Merge backlog: <https://github.com/vladmandic/automatic/pulls>

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

## Random

- Bunch of stuff: <https://pharmapsychotic.com/tools.html>

### Pending Code Updates

- full CUDA tuning section in UI Settings
- improve compatibility with some 3rd party extensions
- improve exif/pnginfo metadata parsing
