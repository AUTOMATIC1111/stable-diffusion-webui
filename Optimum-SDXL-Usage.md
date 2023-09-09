Here's a quick listing of things to tune for your setup:

## Commandline arguments:

- Nvidia (12gb+) `--xformers`
- Nvidia (8gb) `--medvram-sdxl --xformers`
- Nvidia (4gb) `--lowvram --xformers`


## System:
- Downgrade [Nvidia drivers](https://www.nvidia.com/en-us/geforce/drivers/) to 531 or lower prevent extreme slowdowns for largest pictures.
- Add a pagefile to prevent failure loading weights due to low RAM.
- (Linux) install `tcmalloc`, greatly reducing RAM usage: `sudo apt install --no-install-recommends google-perftools` ([#10117](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10117)).
- Use an SSD for faster load time, especially if a pagefile is required.
- Convert `.safetensors` to `.ckpt` for reduced RAM usage ([#12086](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/12086#issuecomment-1691154698)).

## Model weights:
- Use a VAE that will not need to run in fp32 for increased speed and less VRAM usage: [sdxl_vae.safetensors](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors).
- Use [TAESD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#taesd).
- Use fp16 weights (~7gb) for less RAM usage.