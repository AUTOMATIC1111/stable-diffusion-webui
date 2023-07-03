# Diffusers WiP  

initial support merged into `dev` branch  

    git clone https://github.com/vladmandic/automatic -b dev diffusers
    cd diffusers
    webui --debug --backend diffusers

default sd 1.5 model will be downloaded automatically to `models/Diffusers`

on first startup, disable **controlnet** and **multi-diffusion** extensions as right now they are not compatible with diffusers  

to update repo, do not use `--upgrade` flag, use manual `git pull` instead

## Test

### Standard

- run with `webui --debug --backend original`  
- goal is to test standard workflows (so not diffusers) to ensure there are no regressions  
  so diffusers code can be merged into `master` and we can continue with development there

### Diffusers

- sd 1.5 and sd 2.1 model  
- model downloader: tabs -> models -> hf hub
- txt2img, img2img, inpaint, outpaint, process
- hires fix, restore faces, etc?

### Experimental - don't test yet

- cuda model compile using `reduce overhead` model with or without `fullgraph`
- kandinsky model

## Todo

- lora
- embedding
- safetensors models  
- cleanup logging  
- controlnet extension  
- multidiffusion extension  
- sdxl model  

## Limitations

- extra networks
- controlnet
- multi-diffusion

## Issues

- TBD
