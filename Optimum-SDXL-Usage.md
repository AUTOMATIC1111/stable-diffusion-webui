Here's a quick listing of things to tune for your setup:

## Commandline arguments:

- Nvidia (12gb+) `--xformers`
- Nvidia (8gb) `--medvram-sdxl --xformers`
- Nvidia (4gb) `--lowvram --xformers`
- AMD (4gb) `--lowvram --opt-sub-quad-attention` + TAESD in settings
<details> Both rocm and directml will generate at least 1024x1024 pictures at fp16. However, at full precision, the model fails to load (into 4gb). If your card needs --no-half, try enabling --upcast-sampling instead. </details>

## System:
- (Windows) Downgrade [Nvidia drivers](https://www.nvidia.com/en-us/geforce/drivers/) to 531 or lower.  New drivers cause extreme slowdowns on Windows when generating large images towards your card's maximum vram. \
This important issue is discussed [here](https://github.com/vladmandic/automatic/discussions/1285) and in [#11063](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/11063). \
Symptoms:

    * You see Shared GPU memory usage filling up in Task Manager
    * Your generations that usually take 1-2 min, take 7+ min
    * low vram cards are generating very slowly
 
- Add a pagefile to prevent failure loading weights due to low RAM.
- (Linux) install `tcmalloc`, greatly reducing RAM usage: `sudo apt install --no-install-recommends google-perftools` ([#10117](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10117)).
- Use an SSD for faster load time, especially if a pagefile is required.

## Model weights:
- Use [sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors), a VAE that will not need to run in fp32, for increased speed and less VRAM usage.
- Use [TAESD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#taesd); a VAE that uses drastically less vram at the cost of some quality.