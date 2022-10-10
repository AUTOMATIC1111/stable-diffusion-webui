# auto-sd-krita

> AUTOMATIC1111's webUI + Krita Plugin

![demo image](https://user-images.githubusercontent.com/42513874/194701722-e7a3f7eb-be4a-4f43-93a5-480835c9260f.jpg)

Why use this?

- Optimized focus inpainting workflow allowing multiple models to be used for advanced composition of multiple characters.
- AUTOMATIC1111's webUI works alongside plugin to allow using features not available yet in the plugin's UI without restarting.
- Updated regularly with upstream AUTOMATIC1111's features & fixes.

This repository was originally a fork of [sddebz/stable-diffusion-krita-plugin](https://github.com/sddebz/stable-diffusion-krita-plugin), which is itself a fork of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), the most featureful & performant SD fork yet. The main value adds are fixing the commit history, thoroughly refactoring the plugin code for future development, and maintenance.

## Quick Jump

- Full Installation & Workflow Tutorial! (Coming Soon...)
- [Installation Guide](https://github.com/Interpause/auto-sd-krita/wiki/Install-Guide)
  - Similar in difficulty to AUTOMATIC1111 + 1 extra step
- [Usage Guide](https://github.com/Interpause/auto-sd-krita/wiki/Usage-Guide)
- [Features](https://github.com/Interpause/auto-sd-krita/wiki/Features)
- [TODO](https://github.com/Interpause/auto-sd-krita/wiki/TODO)
- [Contribution Guide](https://github.com/Interpause/auto-sd-krita/wiki/Contribution-Guide)

Usage & Workflow Demo:

[![Youtube Video](http://img.youtube.com/vi/nP8MuRwcDN8/0.jpg)](https://youtu.be/nP8MuRwcDN8 "Inpaint like a pro with Stable Diffusion! auto-sd-krita workflow guide")

## FAQ

Q. How hard is it to install?

A. It is basically AUTOMATIC1111 + a few extra steps. aka almost one-click install.

<hr/>

Q. How hard is it to run?

A. The launch process is exactly the same as AUTOMATIC1111, to the point the Gradio webUI functions normally.

<hr/>

Q. Outpainting?

A. No outpainting MK2 yet, but nothing stopping you from doing basic outpainting. 1) expand canvas 2) scribble in blank area 3) img2img on blank area + some of image.

<hr/>

Q. Is the model loaded into memory twice?

A. No, it shares the same backend. Both the Krita plugin and webUI can be used concurrently.

<hr/>

Q. How can you commit to updating regularly?

A. The plugin builds on top the internal API without modifying it, and good documentation practices facilitate adapting to upstream changes and bugfixing.

<hr/>

Q. Will it work with other Krita plugins?

A. Unfortunately no, all plugins so far have different APIs.

## Credits

- [@sddebz](https://github.com/sddebz) for writing the original backend API and Krita plugin while keeping the Gradio webUI functionality intact.
- Stable Diffusion - <https://github.com/CompVis/stable-diffusion>, <https://github.com/CompVis/taming-transformers>
- k-diffusion - <https://github.com/crowsonkb/k-diffusion.git>
- GFPGAN - <https://github.com/TencentARC/GFPGAN.git>
- CodeFormer - <https://github.com/sczhou/CodeFormer>
- ESRGAN - <https://github.com/xinntao/ESRGAN>
- SwinIR - <https://github.com/JingyunLiang/SwinIR>
- LDSR - <https://github.com/Hafiidz/latent-diffusion>
- Ideas for optimizations - <https://github.com/basujindal/stable-diffusion>
- Doggettx - Cross Attention layer optimization - <https://github.com/Doggettx/stable-diffusion>, original idea for prompt editing.
- Rinon Gal - Textual Inversion - <https://github.com/rinongal/textual_inversion> (we're not using his code, but we are using his ideas).
- Idea for SD upscale - <https://github.com/jquesnelle/txt2imghd>
- Noise generation for outpainting mk2 - <https://github.com/parlance-zz/g-diffuser-bot>
- CLIP interrogator idea and borrowing some code - <https://github.com/pharmapsychotic/clip-interrogator>
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)

## License

MIT for the Krita Plugin backend server & frontend plugin. Code has been nearly completely rewritten compared to original plugin by now.
