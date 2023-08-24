Here's a quick listing of things to tune for your setup:

## Commandline arguments:

- (nvidia) 8gb `--medvram-sdxl --xformers`
- (nvidia) 12gb+ `--xformers`
- (nvidia) 4gb `--lowvram --xformers`


## System:
- downgrade nvidia drivers to 531 or lower prevent extreme slowdowns for largest pictures
- add a pagefile to prevent failure to load weights due to low cpu ram
- (Linux) install tcmalloc - greatly reducing ram usage: `sudo apt install --no-install-recommends google-perftools` [#10117](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10117)
- use an SSD, for faster loadtime, especially if a pagefile is required
- converting `.safetensors` to `.ckpt` for reduced ram usage [#12086](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/12086#issuecomment-1691154698)

## Model weights:

- use vae that will not need to run in fp32  for increased speed and less vram usage: [sdxl_vae.safetensors](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors)
- use fp16 (~7gb) weights for less cpu ram usage