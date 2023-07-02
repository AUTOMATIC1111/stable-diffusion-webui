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
  models can be downloaded from huggingface hub  
  but focus on default model for now and i'll add downloader soon  
- lora, textual inversion  
  only loras/textual-inversions downloaded from huggingface hub are supported for now  
  i'll add standard safetensors soon  
- txt2img, img2img, inpaint, outpaint, process

### Experimental

- cuda model compile using `reduce overhead` model with and without `fullgraph`
- kandinsky model

## Todo

- enable loading of safetensors models  
- cleanup logging  
- search&download models from hfhub  
- controlnet extension  
- multidiffusion extension  
- sdxl model  

## Issues

- TBD
