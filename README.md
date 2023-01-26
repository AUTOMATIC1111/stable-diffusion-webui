# Stable Diffusion - Automatic

*Heavily opinionated custom fork of* <https://github.com/AUTOMATIC1111/stable-diffusion-webui>  

![](ui-screenshot.jpg)

<br>

## Notes

Fork is as close as up-to-date with origin as time allows  
All code changes are merged upstream whenever possible  

Fork adds extra functionality:
- Ships with additional **extensions**  
  e.g. `System Info`, `Steps Animation`, etc.  
- Ships with set of **CLI** tools that rely on *SD API* for execution:  
  e.g. `generate`, `train`, `bench`, etc.  
  [Full list](<cli/>)

Simplified start script: `automatic.sh`  
*Existing `webui.sh` still exists for backward compatibility, fresh installs to auto-install dependencies, etc.*  

> ./automatic.sh  

- Start in default mode with optimizations enabled

> ./automatic.sh env  

- Print env info and exit  
  Example:

      Version: c07487a Tue Jan 24 08:04:31 2023 -0500
      Platform: Ubuntu 22.04.1 LTS 5.15.79.1-microsoft-standard-WSL2 x86_64
      Python 3.10.6
      Torch: 2.0.0.dev20230118+cu118 CUDA: 11.8 cuDNN: 8700 GPU: NVIDIA GeForce RTX 3060 Arch: (8, 6)

<<<<<<< HEAD
- Stable Diffusion - https://github.com/CompVis/stable-diffusion, https://github.com/CompVis/taming-transformers
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- GFPGAN - https://github.com/TencentARC/GFPGAN.git
- CodeFormer - https://github.com/sczhou/CodeFormer
- ESRGAN - https://github.com/xinntao/ESRGAN
- SwinIR - https://github.com/JingyunLiang/SwinIR
- Swin2SR - https://github.com/mv-lab/swin2sr
- LDSR - https://github.com/Hafiidz/latent-diffusion
- MiDaS - https://github.com/isl-org/MiDaS
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Cross Attention layer optimization - Doggettx - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- Cross Attention layer optimization - InvokeAI, lstein - https://github.com/invoke-ai/InvokeAI (originally http://github.com/lstein/stable-diffusion)
- Sub-quadratic Cross Attention layer optimization - Alex Birch (https://github.com/Birch-san/diffusers/pull/1), Amin Rezaei (https://github.com/AminRezaei0x443/memory-efficient-attention)
- Textual Inversion - Rinon Gal - https://github.com/rinongal/textual_inversion (we're not using his code, but we are using his ideas).
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Idea for Composable Diffusion - https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
- xformers - https://github.com/facebookresearch/xformers
- DeepDanbooru - interrogator for anime diffusers https://github.com/KichangKim/DeepDanbooru
- Sampling in float32 precision from a float16 UNet - marunine for the idea, Birch-san for the example Diffusers implementation (https://github.com/Birch-san/diffusers-play/tree/92feee6)
- Instruct pix2pix - Tim Brooks (star), Aleksander Holynski (star), Alexei A. Efros (no star) - https://github.com/timothybrooks/instruct-pix2pix
- Security advice - RyotaK
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)
=======
> ./automatic.sh install  

- Install requirements and exit

> ./automatic.sh public  

- Start with listen on public IP with authentication enabled

> ./automatic.sh clean  

- Start with all optimizations disabled  
  Use this for troubleshooting  

<br>

## Differences

Fork does differ in few things:
- Drops compatibility with `python` **3.7** and requires **3.9**  
- Updated **Python** libraries to latest known compatible versions  
  e.g. `accelerate`, `transformers`, `numpy`, etc.  
- Includes opinionated **System** and **Options** configuration  
  e.g. `samplers`, `upscalers`, etc.  
- Includes reskinned **UI**  
  Black and orange dark theme with fixed width options panels and larger previews  
- Includes **SD2** configuration files  
- Uses simplified folder structure  
  e.g. `/train`, `/outputs/*`  
- Modified training templates  

Only Python library which is not auto-updated is `PyTorch` itself as that is very system specific  
For some Torch optimizations notes, see Wiki

Fork is compatible with regular **PyTorch 1.13** as well as pre-release of **PyTorch 2.0**  
See [Wiki](https://github.com/vladmandic/automatic/wiki) for **Torch** optimization notes


<br>

## Docs

Everything is in [Wiki](https://github.com/vladmandic/automatic/wiki)  
Except my current [TODO](TODO.md)  
>>>>>>> 92b4fbe (update requirements)
