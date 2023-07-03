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

## Limitations

even if extensions are not supported, runtime errors are never nice  
will need to handle in the code before we get out of alpha

- controlnet
  `sd_model.model?.diffusion_model?`
- multi-diffusion  
  `sd_model.first_stage_model?.encoder?`
- lycoris
  `lyco_patch_lora`

## Issues

- TBD

## Notes for HF

- removed `quicksettings` alternative completely
- added simple model downloader in ui: *tabs -> models -> huggingface*
- redone **textual inversion** support, core is now in `modules/textual_inversion/textual_inversion.py:load_diffusers_embedding()`  
  the point is that sdnext pre-loads all compatible embeddings on model load so they are available in prompt context
- added support for diffuser models in **safetensors/ckpt** format  
  btw, when i use: `diffusers.StableDiffusionPipeline.from_ckpt`  
  first time it downloads something - what is that?
  > Downloading (…)lve/main/config.json: 4.55k  
  > Downloading pytorch_model.bin: 1.22G  
  and in general, loading safetensors model is quite slow, is that expected?  
  for example, 2sec vs 18sec  
- in `modules/modelloader.py:download_diffusers_model()` i get unknown property for `hf.model_info(hub_id).cardData`
  can you double-check if this is linter issue or actual problem?
- redone **lora** support, core is now in `modules/lora_diffusers.py`  
- question on `pipe.load_lora_weights`
  does it support loading multiple loras? i don't see any notes on that in docs
  also, lora strength is specified using `cross_attention_kwargs={"scale": x}` during pipeline execution  
  which means if there are multiple loras, they all have the same strength?
