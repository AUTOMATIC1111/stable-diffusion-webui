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
*Existing `webui.sh`/`webui.bat` still exist for backward compatibility, fresh installs to auto-install dependencies, etc.*  

> ./automatic.sh  

- Start in default mode with optimizations enabled  
  Additionally print environment info during startup  
  Example:  

      Version: c07487a Tue Jan 24 08:04:31 2023 -0500
      Platform: Ubuntu 22.04.1 LTS 5.15.79.1-microsoft-standard-WSL2 x86_64
      Python 3.10.6
      Torch: 2.0.0.dev20230118+cu118 CUDA: 11.8 cuDNN: 8700 GPU: NVIDIA GeForce RTX 3060 Arch: (8, 6)

> ./automatic.sh public  

- Start with listen on public IP with authentication enabled

> ./automatic.sh clean  

- Start with all optimizations disabled  
  Use this for troubleshooting  

<br>  

## Install

1. Install `PyTorch` first
2. Clone and initialize repository

> git clone --depth 1 https://github.com/vladmandic/automatic  
> git submodule update --init --recursive  

3. Install dependencies

> ./automatic.sh install  

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

## Scripts

This repository comes with a large collection of scripts that can be used to process inputs, train, generate, and benchmark models  

As well as number of auxiliary scripts that do not rely on **WebUI**, but can be used for end-to-end solutions such as extract frames from videos, etc.  

For full details see [Docs](cli/README.md)

<br>

## Docs

- Scripts are in [Scripts](cli/README.md)  
- Everything else is in [Wiki](https://github.com/vladmandic/automatic/wiki)  
- Except my current [TODO](TODO.md)  
