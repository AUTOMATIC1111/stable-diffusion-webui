# Diffusers

## Install

initial support merged into `dev` branch  

- download from branch and start as normal:
  > git clone https://github.com/vladmandic/automatic -b dev diffusers  
  > cd diffusers  
  > webui --debug --backend diffusers  

- to go back to standard execution pipeline, start with  
  > webui --debug --backend original

- To update repo, do not use `--upgrade` flag, use manual `git pull` instead

## Notes

All notes have moved to [Wiki page](https://github.com/vladmandic/automatic/wiki/Diffusers)

## TODO

- VAE
  > vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")  
  > pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)  
- Refiner handler with medvram/lowvram
