# TODO

## Issues

Stuff to be fixed...

- Fix integration with <https://github.com/kohya-ss/sd-webui-additional-networks>
- Fix integration with <https://github.com/deforum-art/deforum-for-automatic1111-webui>
- Reconnect UI to ops in progress on browser restart  
- Create new GitHub hooks/actions for CI/CD  
- Remove `models` from git repo

## Features

Stuff to be added...

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

- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/7595>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/8608>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/8611>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/8665>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/8741>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/8742>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9128>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9134>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9155>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9211>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9212>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9219>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9227>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9249>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9256>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9258>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9295>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9312>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9314>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9315>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9319>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9392>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9407>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9451>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9491>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9504>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9513>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9542>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9592>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9628>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9669>
- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9677>

Complete:

- <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/8945>

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

### Update

- reconnect ui to active session on browser restart  
  this is one of most frequently asked for items, finally figured it out  
  works for text and image generation, but not for process as there is no progress bar reported there to start with  
- force unload `xformers` when not used  
  improves compatibility with AMD/M1 platforms  
- add `styles.csv` to UI settings to allow customizing path  
- add `--skip-git` to cmd flags for power users that want  
  to skip all git checks and operations and perform manual updates
- add `--disable-queue` to cmd flags that disables Gradio queues (experimental)
  this forces it to use HTTP instead of WebSockets and can help on unreliable network connections  
- set scripts & extensions loading priority and allow custom priorities  
  fixes random extension issues:  
  `ScuNet` upscaler dissapearing, `Additional Networks` not showing up on XYZ axis, etc.
- improve html loading order
- remove some `asserts` causing runtime errors and replace with user-friendly messages
- update README.md
