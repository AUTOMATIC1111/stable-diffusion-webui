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
- new schedulers
- settings -> schedulers

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

- new dependency hell (not diffuser related)?

## Notes for HF

- removed `quicksettings` alternative completely
- added simple model downloader in ui: *tabs -> models -> huggingface*  
- attempting to download gated model without access token results in model/refs/commits not found instead of access denied  
  this is not very user friendly, it should be handled in the code  
- redone **textual inversion** support, core is now in `modules/textual_inversion/textual_inversion.py:load_diffusers_embedding()`  
  the point is that sdnext pre-loads all compatible embeddings on model load so they are available in prompt context
- redone **lora** support, core is now in `modules/lora_diffusers.py`  
- added support for diffuser models in **safetensors/ckpt** format  
- when i use `diffusers.StableDiffusionPipeline.from_ckpt`  
  first time it downloads something - what is that? (could it be a default safety checker?)
  > Downloading (…)lve/main/config.json: 4.55k  
  > Downloading pytorch_model.bin: 1.22G  
- loading safetensors model is very slow  
  for example, 2sec without diffusers and 16sec with diffusers
- in `modules/modelloader.py:download_diffusers_model()` i get unknown property for `hf.model_info(hub_id).cardData`
  can you double-check if this is linter issue or actual problem?
- question on `pipe.load_lora_weights`
  does it support loading multiple loras? i don't see any notes on that in docs
  also, lora strength is specified using `cross_attention_kwargs={"scale": x}` during pipeline execution  
  which means if there are multiple loras, they all have the same strength?
- **deepfloyd** failures:
  > /home/disty/Apps/automatic/venv/lib/python3.10/site-packages/diffusers/configuration_utils.py:138 in __getattr__  
  > AttributeError: 'DDPMScheduler' object has no attribute 'name  
- question how do diffusers handle standard 75 token limit for sd?  
- diffusers `convert_from_ckpt.py` uses fixed `print` statements so its not possible to control its output to console  
  it should use `logging` instead. in general, using `print` is bad idea  
  for example, it very annoyingly logs this every time `StableDiffusionPipeline.from_ckpt` is used:
  > global_step key not found in model  
  > Checkpoint /home/vlado/dev/automatic/models/Stable-diffusion/best/absolutereality_v1.safetensors has both EMA and non-EMA weights.  
  > In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag.  
- do you have plans to implement [Restart](https://github.com/vladmandic/automatic/issues/1537) sampler in diffusers?
- scheduler config is really difficult to work with as its not possible to see which params each scheduler defines ahead of time and if passing params it doesn't have, it will result in runtime error

## Update

- sortable models table in downloader ui  
- system info tab -> benchmark is now working  
- recommended scheduler: `deis`  
- `channels_last` and `cudnn_benchmark` now apply to diffusers  
- new settings section for diffusers fine-tuning  
- fixed missed call to `devices.set_cuda_params`
- had to reduce number of supported schedulers by a lot until i add param checking for the rest  
  issue is that diffusers have completely different params for schedulers than a111, but passing unknown param causes runtime error
  previously params were not passed at all, so you couldn't even use anything other than default scheduler (although ui showed you were)
- fixed "it looks like the config file at 'xxx.safetensors' is not a valid JSON file"  
  this is also related to schedulers as diffusers are trying to read default scheduler config from model itself, but that doesn't exist for safetensors  
