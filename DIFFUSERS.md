# Diffusers

## Install

initial support merged into `dev` branch  

- first download and start as normal:
  > git clone https://github.com/vladmandic/automatic -b dev diffusers  
  > cd diffusers  
  > webui --debug --backend original  

- then upgrade diffusers to unreleased version and switch to using diffusers
  > pip install --upgrade git+https://github.com/huggingface/diffusers  
  > webui --debug --quick --backend diffusers  

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
