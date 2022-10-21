# auto-sd-krita

> AUTOMATIC1111's webUI + Krita Plugin

![demo image](https://user-images.githubusercontent.com/42513874/194701722-e7a3f7eb-be4a-4f43-93a5-480835c9260f.jpg)

Why use this?

- Optimized focus inpainting workflow allowing multiple models to be used for advanced composition of multiple characters.
- AUTOMATIC1111's webUI works alongside plugin to allow using features not available yet in the plugin's UI without restarting.
- Updated regularly with upstream AUTOMATIC1111's features & fixes.

This repository was originally a fork of [sddebz/stable-diffusion-krita-plugin](https://github.com/sddebz/stable-diffusion-krita-plugin), which is itself a fork of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), the most featureful & performant SD fork yet. The main value adds are fixing the commit history, thoroughly refactoring the plugin code for future development, and maintenance.

## Quick Jump

- Full Installation & Workflow Tutorial Video! (Coming Soon...)
- [Installation Guide](https://github.com/Interpause/auto-sd-krita/wiki/Install-Guide)
  - Similar in difficulty to AUTOMATIC1111 + 1 extra step
- [Test out auto-sd-krita with Existing AUTOMATIC1111 Install!](https://github.com/Interpause/auto-sd-krita/wiki/Quick-Switch-Using-Existing-AUTOMATIC1111-Install)
- [Usage Guide](https://github.com/Interpause/auto-sd-krita/wiki/Usage-Guide)
- [Features](https://github.com/Interpause/auto-sd-krita/wiki/Features)
- [TODO](https://github.com/Interpause/auto-sd-krita/wiki/TODO)
- [Contribution Guide](https://github.com/Interpause/auto-sd-krita/wiki/Contribution-Guide)

Usage & Workflow Demo:

[![Youtube Video](http://img.youtube.com/vi/nP8MuRwcDN8/0.jpg)](https://youtu.be/nP8MuRwcDN8 "Inpaint like a pro with Stable Diffusion! auto-sd-krita workflow guide")

## UI Changelog

- No need to manually hide inpainting layer anymore; It will be auto-hidden.
- Color correction can be toggled separately for img2img/inpainting.
- Status bar:
  - In middle of page to be more visible even when scrolling.
  - Warning when using features with no document open.
- Inpaint is now the default tab.

## FAQ

Q: How hard is it to install?

A: It is basically AUTOMATIC1111 + a few extra steps. aka almost one-click install.

<hr/>

Q: How hard is it to run?

A: The launch process is exactly the same as AUTOMATIC1111, to the point the Gradio webUI functions normally.

<hr/>

Q: How does the base_size, max_size system work?

A:

It is an alternative to AUTO's highres fix that works for all modes, not just txt2img.

The selection will be resized such that the shorter dimension is base_size. However, if the aforementioned resize causes the longer dimension to exceed max_size, the shorter dimension will be resized to less than base_size. Setting base_size and max_size higher can be used to generate higher resolution images (along with their issues), essentially **disabling the system**, _though it might make sense for img2img mode_.

This is actually smarter than the builtin highres fix + firstphase width/height system. Thank the original plugin writer, @sddebz, for writing this.

<hr/>

Q: Outpainting?

A: No outpainting MK2 yet, but nothing stopping you from doing basic outpainting. 1) expand canvas 2) scribble in blank area 3) img2img on blank area + some of image.

<hr/>

Q: Is the model loaded into memory twice?

A: No, it shares the same backend. Both the Krita plugin and webUI can be used concurrently.

<hr/>

Q: How can you commit to updating regularly?

A: The plugin builds on top the internal API without modifying it, and good documentation practices facilitate adapting to upstream changes and bugfixing.

<hr/>

Q: Will it work with other Krita plugin backends?

A: Unfortunately no, all plugins so far have different APIs. The official API is coming soon though...

## Credits

- [@sddebz](https://github.com/sddebz) for writing the original backend API and Krita plugin while keeping the Gradio webUI functionality intact.
- Stable Diffusion - https://github.com/CompVis/stable-diffusion, https://github.com/CompVis/taming-transformers
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- GFPGAN - https://github.com/TencentARC/GFPGAN.git
- CodeFormer - https://github.com/sczhou/CodeFormer
- ESRGAN - https://github.com/xinntao/ESRGAN
- SwinIR - https://github.com/JingyunLiang/SwinIR
- Swin2SR - https://github.com/mv-lab/swin2sr
- LDSR - https://github.com/Hafiidz/latent-diffusion
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Doggettx - Cross Attention layer optimization - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- InvokeAI, lstein - Cross Attention layer optimization - https://github.com/invoke-ai/InvokeAI (originally http://github.com/lstein/stable-diffusion)
- Rinon Gal - Textual Inversion - https://github.com/rinongal/textual_inversion (we're not using his code, but we are using his ideas).
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Idea for Composable Diffusion - https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
- xformers - https://github.com/facebookresearch/xformers
- DeepDanbooru - interrogator for anime diffusers https://github.com/KichangKim/DeepDanbooru
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)

## License

MIT for the Krita Plugin backend server & frontend plugin. Code has been nearly completely rewritten compared to original plugin by now.
