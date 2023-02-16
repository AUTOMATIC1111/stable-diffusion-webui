# Stable Diffusion - Automatic

*Heavily opinionated custom fork of* <https://github.com/AUTOMATIC1111/stable-diffusion-webui>  

![](ui-screenshot.jpg)

<br>

## Notes

Fork is as close as up-to-date with origin as time allows  
All code changes are merged upstream whenever possible  

Fork adds extra functionality:
- New skin and UI layout  
- Ships with additional **extensions**  
  e.g. `System Info`, `Steps Animation`, etc.  
- Ships with set of **CLI** tools that rely on *SD API* for execution:  
  e.g. `generate`, `train`, `bench`, etc.  
  [Full list](<cli/>)

Simplified start script: `automatic.sh`  
*Existing `webui.sh`/`webui.bat` still exist for backward compatibility, fresh installs to auto-install dependencies, etc.*  

> ./automatic.sh  

- Start in default mode with optimizations enabled  
  Additionally print environment info during startup  
  Example:  

      Version: b0b326f3 Wed Feb 15 09:07:04 2023 -0500
      Repository: https://github.com/vladmandic/automatic
      Last Merge: Sun Feb 5 07:03:27 2023 -0500 Merge pull request #35 from AUTOMATIC1111/master
      Platform: Ubuntu 22.04.1 LTS 5.15.83.1-microsoft-standard-WSL2 x86_64
      nVIDIA: NVIDIA GeForce RTX 3060, 528.49
      Python: 3.10.6 Torch: 2.0.0.dev20230211+cu118 CUDA: 11.8 cuDNN: 8700 GPU: NVIDIA GeForce RTX 3060 Arch: (8, 6)

> ./automatic.sh public  

- Start with listen on public IP with authentication enabled

> ./automatic.sh clean  

- Start with all optimizations disabled  
  Use this for troubleshooting  

> ./automatic.sh install

- Installs and refreshes:  
  dependencies, submodules, extensions  

<br>  

## Install

1. Install `PyTorch` first
2. Clone and initialize repository

> git clone --depth 1 https://github.com/vladmandic/automatic  
> cd automatic  
> ./automatic.sh install  

      SD server: install
      Installing general requirements
      Installing versioned requirements
      Updating submodules
      Modules:
      - 6c76a48 Mon Feb 13 00:03:00 2023 -0800 https://github.com/mcmonkeyprojects/sd-dynamic-thresholding
      - a528cd5 Tue Jan 31 07:57:07 2023 -0500 https://github.com/vladmandic/sd-extension-aesthetic-scorer
      - 7cf0e3a Tue Feb 7 07:39:40 2023 -0500 https://github.com/vladmandic/sd-extension-steps-animation
      - b5d8e6a Thu Feb 9 15:25:18 2023 -0500 https://github.com/vladmandic/sd-extension-system-info
      - 7a998ed Wed Feb 8 07:21:52 2023 -0500 https://github.com/Akegarasu/sd-webui-model-converter
      - 0a5c897 Thu Feb 16 11:08:00 2023 +0100 https://github.com/yownas/seed_travel
      - c8efd35 Mon Feb 13 21:51:25 2023 +0100 https://github.com/AlUlkesh/stable-diffusion-webui-images-browser
      - 14d7b24 Thu Feb 16 22:35:47 2023 +0900 https://github.com/kohya-ss/sd-scripts
      - b351828 Tue Feb 14 11:47:19 2023 -0500 https://github.com/vladmandic/automatic.wiki
      Updating extensions
      Extensions:
      - e5b773a Sat Feb 11 19:38:18 2023 +0500 https://github.com/klimaleksus/stable-diffusion-webui-embedding-merge
      - 0f3f699 Fri Dec 9 11:50:47 2022 +0800 https://github.com/yfszzx/stable-diffusion-webui-inspiration

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
- Built-in `LoRA` training  
- Built-in `Custom Diffusion` training  

Only Python library which is not auto-updated is `PyTorch` itself as that is very system specific  
For some Torch optimizations notes, see Wiki

Fork is compatible with regular **PyTorch 1.13** as well as pre-release of **PyTorch 2.0**  
See [Wiki](https://github.com/vladmandic/automatic/wiki) for **Torch** optimization notes

<br>

## Scripts

This repository comes with a large collection of scripts that can be used to process inputs, train, generate, and benchmark models  

As well as number of auxiliary scripts that do not rely on **WebUI**, but can be used for end-to-end solutions such as extract frames from videos, etc.  

For full details see [Docs](cli/README.md)

<br>

## Docs

- Scripts are in [Scripts](cli/README.md)  
- Everything else is in [Wiki](https://github.com/vladmandic/automatic/wiki)  
- Except my current [TODO](TODO.md)  
