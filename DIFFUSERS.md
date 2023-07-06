# Diffusers WiP  

initial support merged into `dev` branch  

    git clone https://github.com/vladmandic/automatic -b dev diffusers
    cd diffusers
    webui --debug --backend diffusers

default sd 1.5 model will be downloaded automatically to `models/Diffusers`

on first startup, disable **controlnet** and **multi-diffusion** extensions as right now they are not compatible with diffusers  
lora support is not compatible with setting `Use LyCoris handler for all Lora types`, make sure its disabled

to update repo, do not use `--upgrade` flag, use manual `git pull` instead

## Test

### Standard

goal is to test standard workflows (so not diffusers) to ensure there are no regressions  
so diffusers code can be merged into `master` and we can continue with development there

- run with `webui --debug --backend original`  

### Diffusers

whats implemented so far?

- new scheduler: deis
- simple model downloader for huggingface models: tabs -> models -> hf hub  
- use huggingface models  
- extra networks ui  
- use safetensor models with diffusers backend  
- lowvram and medvram equivalents for diffusers
- standard workflows:
  - txt2img, img2img, inpaint, outpaint, process  
  - hires fix, restore faces, etc?  
- textual inversion  
  yes, this applies to standard embedddings, don't need ones from huggingface  
- lora  
  yes, this applies to standard loras, don't need ones from huggingface  
  but seems that diffuser lora support is somewhat limited, so quite a few loras may not work  
  you should see which lora loads without issues in console log  
- system info tab with updated information  
- kandinsky model  
  works for me  

### Experimental

- cuda model compile
  in settings -> compute settings  
  diffusers recommend `reduce overhead`, but other methods are available as well  
  it seems that fullgraph is possible (with sufficient vram) when using diffusers  
- deepfloyd  
  in theory it should work, but its 20gb model so cant test it just yet  
  note that access is gated, so you'll need to download using your huggingface credentials  
  (you can still do it from sdnext ui, just need access token)  

## Todo

- sdxl model  
- no idea if sd21 works out-of-the-box
- hires fix?

## Limitations

even if extensions are not supported, runtime errors are never nice  
will need to handle in the code before we get out of alpha

- lycoris  
  `lyco_patch_lora`
- controlnet
  > sd_model.model?.diffusion_model?
- multi-diffusion  
  > sd_model.first_stage_model?.encoder?
- dynamic-thresholding
  > AttributeError: 'DiffusionSampler' object has no attribute 'model_wrap_cfg'

## Performance

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

- Performance is measured for `batch-size` 1, 2, 4, 8 16
- Test environment:
  - nVidia RTX 3060 GPU
  - Torch 2.1-nightly with CUDA 12.1
  - Cross-optimization: SDP
- All being equal, diffussers seem to:
  - Use slightly less RAM and more VRAM
  - Have highly efficient medvram/lowvram equivalents which don't loose a lot of performance  
  - Faster on smaller batch sizes, slower on larger batch sizes  
