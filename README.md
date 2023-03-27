# Stable Diffusion - Automatic

*Heavily opinionated custom fork of* <https://github.com/AUTOMATIC1111/stable-diffusion-webui>  

Fork is as close as up-to-date with origin as time allows  
All code changes are merged upstream whenever possible  

![screenshot](ui-screenshot.jpg)

<br>

## Notes

### Fork does differ in few things

- New error and exception handlers  
- Updated **Python** libraries to latest known compatible versions  
  e.g. `accelerate`, `transformers`, `numpy`, etc.  
- Includes opinionated **System** and **Options** configuration  
  e.g. `samplers`, `upscalers`, etc.  
- Does not rely on `Accelerate` as it only affects distributed systems  
- Optimized startup  
  Gradio web server will be initialized much earlier which model load is done in the background  
  Faster model loading plus ability to fallback on corrupt models  
- Includes **SD2** configuration files  
- Uses simplified folder structure  
  e.g. `/train`, `/outputs/*`, `/models/*`, etc.  
- Enhanced training templates  
- Built-in `LoRA`, `LyCORIS`, `Custom Diffusion`, `Dreambooth` training  

### User Interface

- Includes updated **UI**: reskinned and reorganized  
  Black and orange dark theme with fixed width options panels and larger previews  

### Optimizations

- Optimized for `Torch` 2.0  
- Runs with `SDP` memory attention enabled by default if supported by system  
  Fallback to `XFormers` if SDP is not supported  
  If either `SDP` or `XFormers` are not supported, falls back to usual cmd line arguments  

### Removed

- Drops compatibility with `python` **3.7** and requires **3.9**  
  Recommended is **Python 3.10**  
  Note that **Python 3.11** or **3.12** are NOT supported  
- Drops localizations  
- Drops automated tests  

### Integrated CLI/API tools

Fork adds extra functionality:

- New skin and UI layout  
- Ships with set of **CLI** tools that rely on *SD API* for execution:  
  e.g. `generate`, `train`, `bench`, etc.  
  [Full list](<cli/>)

### Integrated Extensions

- [System Info](https://github.com/vladmandic/sd-extension-system-info)
- [ControlNet](https://github.com/Mikubill/sd-webui-controlnet)
- [Image Browser](https://github.com/AlUlkesh/stable-diffusion-webui-images-browser)
- [LORA](https://github.com/kohya-ss/sd-scripts) *(both training and inference)*
- [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) *(both training and inference)*
- [Model Converter](https://github.com/Akegarasu/sd-webui-model-converter)
- [CLiP Interrogator](https://github.com/pharmapsychotic/clip-interrogator-ext)
- [Dynamic Thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding)
- [Steps Animation](https://github.com/vladmandic/sd-extension-steps-animation)
- [Seed Travel](https://github.com/yownas/seed_travel)

*Note*: Extensions are automatically updated to latest version on `install`

<br>

### Start Script

Simplified start script: `automatic.sh`  
*Existing `webui.sh`/`webui.bat` scripts still exist for backward compatibility*  

> ./automatic.sh  

Start in default mode with optimizations enabled  

      Stable Diffusion server: optimized
      Version: a4d00060 Sun Mar 26 10:28:05 2023 -0400
      Repository: https://github.com/vladmandic/automatic
      Platform: Ubuntu 22.04.2 LTS 5.15.90.1-microsoft-standard-WSL2 x86_64
      Installing requirements for Web UI
      Launching Web UI with arguments: --cors-allow-origins=http://127.0.0.1:7860 --ckpt models/v1-5-pruned-emaonly.safetensors
      Torch 2.0.0+cu118 CUDA 11.8 cuDNN 8700
      GPU NVIDIA GeForce RTX 3060 VRAM 12288 Arch (8, 6) Cores 28
      Running on local URL:  http://127.0.0.1:7860
      Loading weights: models/v1-5-pruned-emaonly.safetensors ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/4.3 GB -:--:--
      Creating model from config: /home/vlado/dev/automatic/configs/v1-inference.yaml
      DiffusionWrapper has 859.52 M params.
      Loading weights: models/VAE/vae-ft-mse-840000-ema-pruned.ckpt ━━━━━━━━━━━━━━━━━━━━━━━ 0.0/334.7 MB -:--:--
      Applying scaled dot product cross attention optimization.
      Textual inversion embeddings loaded(2): ti-mia, ti-vlado
      Model loaded in 1.6s (load weights: 0.1s, create model: 0.3s, apply weights: 0.4s, load vae: 0.3s, device move: 0.5s).
      Startup time: 11.8s (import torch: 1.7s, import libraries: 1.0s, list models: 1.9s, load scripts: 1.0s, create ui: 4.4s, load checkpoint: 1.7s).
      Progress 6.55it/s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 0:00:04

> ./automatic.sh clean  

Start with all optimizations disabled  
Use this for troubleshooting  

> ./automatic.sh install

Installs and updates to latest supported version:

- Dependencies
- Fixed sub-repositories
- Extensions
- Sub-modules

Does not update main repository

> ./automatic.sh update

Updates the main repository to the latest version  
Recommended to run `install` after `update` to update dependencies as they may have changed  

> ./automatic.sh help

Print all available options

> ./automatic.sh public  

Start with listen on public IP with authentication enabled  

<br>  

## Install

1. Install `Python`, `Git`  
2. Install `PyTorch`
   See [Wiki](wiki/Torch%20Optimizations.md) for details or TL;DR below  
3. Clone and initialize repository  

> git clone https://github.com/vladmandic/automatic  
> cd automatic  
> ./automatic.sh install  

      SD server: install
      Version: 56f779a9 Sat Feb 25 14:04:19 2023 -0500
      Repository: https://github.com/vladmandic/automatic
      Last Merge: Sun Feb 19 10:11:25 2023 -0500 Merge pull request #37 from AUTOMATIC1111/master
      Installing general requirements
      Installing versioned requirements
      Installing requirements for Web UI
      Updating submodules
      Updating extensions
      Updating wiki
      Detached repos
      Local changes

*Note*: If you're not using `automatic.sh` launcher, install dependencies manually:

> pip -r requirements.txt
> pip -r requirements_versions.txt

<br>

## Other

### Torch

Only Python library which is not auto-updated is `PyTorch` itself as that is very system specific  
Fork is compatible with regular **PyTorch 1.13**, **PyTorch 2.0** as well as pre-releases of **PyTorch** **2.1**  
TL;DR: Install **PyTorch 2.0.0** compiled with **CUDA 11.8**:

> pip install torch torchaudio torchvision triton --force --extra-index-url https://download.pytorch.org/whl/cu118  

See [Wiki](https://github.com/vladmandic/automatic/wiki/Torch-Optimizations) for **Torch** optimization notes

<br>

### Scripts

This repository comes with a large collection of scripts that can be used to process inputs, train, generate, and benchmark models  
As well as number of auxiliary scripts that do not rely on **WebUI**, but can be used for end-to-end solutions such as extract frames from videos, etc.  
For full details see [Docs](cli/README.md)

<br>

### Docs

- Scripts are in [Scripts](cli/README.md)  
- Everything else is in [Wiki](https://github.com/vladmandic/automatic/wiki)  
- Except my current [TODO](TODO.md)  

<br>
