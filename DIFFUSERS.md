# Additional Models

SD.Next includes *experimental* support for additional model pipelines  
This includes support for additional models such as:

- **Stable Diffusion XL**
- **Kandinsky**
- **Deep Floyd IF**
- **Shap-E**

Note that support is *experimental*, do not open [GitHub issues](https://github.com/vladmandic/automatic/issues) for those models  
and instead reach-out on [Discord](https://discord.gg/WqMzTUDC) using dedicated channels  

*This has been made possible by integration of [huggingface diffusers](https://huggingface.co/docs/diffusers/index) library with help of huggingface team!*

## How to

- Install **SD.Next** as usual  
- Start with  
  `webui --backend diffusers`
- To go back to standard execution pipeline, start with  
  `webui --backend original`

## Integration

### Standard workflows  

- **txt2txt**
- **img2img**
- **process**

### Model Access

- For standard SD 1.5 and SD 2.1 models, you can use either  
  standard *safetensor* models or *diffusers* models
- For additional models, you can use *diffusers* models only
- You can download diffuser models directly from [Huggingface hub](https://huggingface.co/)  
  or use built-in model search & download in SD.Next: **UI -> Models -> Huggingface**
- Note that access to some models is gated  
  In which case, you need to accept model EULA and provide your huggingface token  

### Extra Networks

- Lora networks  
- Textual inversions (embeddings)  

Note that Lora and TI need are still model-specific, so you cannot use Lora trained on SD 1.5 on SD-XL  
(just like you couldn't do it on SD 2.1 model) - it needs to be trained for a specific model  

Support for SD-XL training is expected shortly  

### Diffuser Settings

- UI -> Settings -> Diffuser Settings  
  contains additional tunable parameters  

### Samplers

- Samplers (schedulers) are pipeline specific, so when running with diffuser backend, you'll see a different list of samplers
- UI -> Settings -> Sampler Settings shows different configurable parameters depending on backend  
- Recommended sampler for diffusers is **DEIS**

### Other

- Updated **System Info** tab with additional information
- Support for `lowvram` and `medvram` modes  
  Additional tunables are available in UI -> Settings -> Diffuser Settings  
- Support for both default **SDP** and **xFormers** cross-optimizations  
  Other cross-optimization methods are not available  
- **Extra Networks UI** will show available diffusers models  
- **CUDA model compile**  
  UI Settings -> Compute settings  
  Requires GPU with high VRAM  
  Diffusers recommend `reduce overhead`, but other methods are available as well  
  Fullgraph is possible (with sufficient vram) when using diffusers  

## SD-XL Notes

- SD-XL model is designed as two-stage model  
  You can run SD-XL pipeline using just `base` model, but for best results, load both `base` and `refiner` models  
  - `base`: Trained on images with variety of aspect ratios and uses OpenCLIP-ViT/G and CLIP-ViT/L for text encoding  
  - `refiner`: Trained to denoise small noise levels of high quality data and uses the OpenCLIP model  
- If you want to use `refiner` model, it is advised to add `sd_model_refiner` to **quicksettings**  
  in UI Settings -> User Interface
- SD-XL model was trained on **1024px** images  
  You can use it with smaller sizes, but you will likely get better results with SD 1.5 models  
- SD-XL model NSFW filter has been turned off  

## Limitations

- Diffusers do not have callbacks per-step, so any functionality that relies on that will not be available  
  This includes trival but very visible **progress bar**  
- Any extension that requires access to model internals will likely not work when using diffusers backend  
  This for example includes standard extensions such as `ControlNet`, `MultiDiffusion`, `LyCORIS`
- Second-pass workflows such as `hires fix` are not yet implemented (soon)
- Hypernetworks  
- Explit VAE usage (soon)

## Performance

Comparison of original stable diffusion pipeline and diffusers pipeline

| pipeline | performance it/s | memory cpu/gpu |
| --- | --- | --- |
| original | 7.99 / 7.93 / 8.83 / 9.14 / 9.2 | 6.7 / 7.2 |
| original medvram | 6.23 / 7.16 / 8.41 / 9.24 / 9.68 | 8.4 / 6.8 |
| original lowvram | 1.05 / 1.94 / 3.2 / 4.81 / 6.46 | 8.8 / 5.2 |
| diffusers | 9 / 7.4 / 8.2 / 8.4 / 7.0 | 4.3 / 9.0 |
| diffusers medvram | 7.5 / 6.7 / 7.5 / 7.8 / 7.2 | 6.6 / 8.2 |
| diffusers lowvram | 7.0 / 7.0 / 7.4 / 7.7 / 7.8 | 4.3 / 7.2 |
| diffusers with safetensors | 8.9 / 7.3 / 8.1 / 8.4 / 7.1 | 5.9 / 9.0 |

Notes:

- Performance is measured using standard SD 1.5 model
- Performance is measured for `batch-size` 1, 2, 4, 8 16
- Test environment:
  - nVidia RTX 3060 GPU
  - Torch 2.1-nightly with CUDA 12.1
  - Cross-optimization: SDP
- All being equal, diffussers seem to:
  - Use slightly less RAM and more VRAM
  - Have highly efficient medvram/lowvram equivalents which don't loose a lot of performance  
  - Faster on smaller batch sizes, slower on larger batch sizes  

## TODO

initial support merged into `dev` branch  

    git clone https://github.com/vladmandic/automatic -b dev diffusers
    cd diffusers
    webui --debug --backend diffusers

default sd 1.5 model will be downloaded automatically to `models/Diffusers`

on first startup, disable **controlnet** and **multi-diffusion** extensions as right now they are not compatible with diffusers  
lora support is not compatible with setting `Use LyCoris handler for all Lora types`, make sure its disabled

to update repo, do not use `--upgrade` flag, use manual `git pull` instead

- lycoris  
  `lyco_patch_lora`
- controlnet
  > sd_model.model?.diffusion_model?
- multi-diffusion  
  > sd_model.first_stage_model?.encoder?
- dynamic-thresholding
  > AttributeError: 'DiffusionSampler' object has no attribute 'model_wrap_cfg'

- diffusers pipeline in general no sampler per-step callback, its completely opaque inside the pipeline  
  so i'm missing some very basic stuff like progress bar in the ui or ability to generate live preview based on intermediate latents  
- StableDiffusionXLPipeline does not implement `from_ckpt`
- StableDiffusionXLPipeline has long delay after tqdm progress bar finishes and before it returns an image, i assume its vae, but its not a good user-experience
- VAE:  
  > vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")  
  > pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)  
