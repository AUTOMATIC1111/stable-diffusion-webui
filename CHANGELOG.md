## 1.2.1

### Features:
 * add an option to always refer to lora by filenames

### Bug Fixes:
 * never refer to lora by an alias if multiple loras have same alias or the alias is called none
 * fix upscalers disappearing after the user reloads UI
 * allow bf16 in safe unpickler (resolves problems with loading some loras)
 * allow web UI to be ran fully offline
 * fix localizations not working
 * fix error for loras: 'LatentDiffusion' object has no attribute 'lora_layer_mapping'

## 1.2.0

### Features:
 * do not wait for stable diffusion model to load at startup
 * add filename patterns: [denoising]
 * directory hiding for extra networks: dirs starting with . will hide their cards on extra network tabs unless specifically searched for
 * Lora: for the `<...>` text in prompt, use name of Lora that is in the metdata of the file, if present, instead of filename (both can be used to activate lora)
 * Lora: read infotext params from kohya-ss's extension parameters if they are present and if his extension is not active
 * Lora: Fix some Loras not working (ones that have 3x3 convolution layer)
 * Lora: add an option to use old method of applying loras (producing same results as with kohya-ss)
 * add version to infotext, footer and console output when starting
 * add links to wiki for filename pattern settings
 * add extended info for quicksettings setting and use multiselect input instead of a text field

### Minor:
 * gradio bumped to 3.29.0
 * torch bumped to 2.0.1
 * --subpath option for gradio for use with reverse proxy
 * linux/OSX: use existing virtualenv if already active (the VIRTUAL_ENV environment variable)
 * possible frontend optimization: do not apply localizations if there are none
 * Add extra `None` option for VAE in XYZ plot
 * print error to console when batch processing in img2img fails
 * create HTML for extra network pages only on demand
 * allow directories starting with . to still list their models for lora, checkpoints, etc
 * put infotext options into their own category in settings tab
 * do not show licenses page when user selects Show all pages in settings

### Extensions:
 * Tooltip localization support
 * Add api method to get LoRA models with prompt

### Bug Fixes:
 * re-add /docs endpoint
 * fix gamepad navigation
 * make the lightbox fullscreen image function properly
 * fix squished thumbnails in extras tab
 * keep "search" filter for extra networks when user refreshes the tab (previously it showed everthing after you refreshed)
 * fix webui showing the same image if you configure the generation to always save results into same file
 * fix bug with upscalers not working properly
 * Fix MPS on PyTorch 2.0.1, Intel Macs
 * make it so that custom context menu from contextMenu.js only disappears after user's click, ignoring non-user click events
 * prevent Reload UI button/link from reloading the page when it's not yet ready
 * fix prompts from file script failing to read contents from a drag/drop file


## 1.1.1
### Bug Fixes:
 * fix an error that prevents running webui on torch<2.0 without --disable-safe-unpickle

## 1.1.0
### Features:
 * switch to torch 2.0.0 (except for AMD GPUs)
 * visual improvements to custom code scripts
 * add filename patterns: [clip_skip], [hasprompt<>], [batch_number], [generation_number]
 * add support for saving init images in img2img, and record their hashes in infotext for reproducability
 * automatically select current word when adjusting weight with ctrl+up/down
 * add dropdowns for X/Y/Z plot
 * setting: Stable Diffusion/Random number generator source: makes it possible to make images generated from a given manual seed consistent across different GPUs
 * support Gradio's theme API
 * use TCMalloc on Linux by default; possible fix for memory leaks
 * (optimization) option to remove negative conditioning at low sigma values #9177
 * embed model merge metadata in .safetensors file
 * extension settings backup/restore feature #9169
 * add "resize by" and "resize to" tabs to img2img
 * add option "keep original size" to textual inversion images preprocess
 * image viewer scrolling via analog stick
 * button to restore the progress from session lost / tab reload

### Minor:
 * gradio bumped to 3.28.1
 * in extra tab, change extras "scale to" to sliders
 * add labels to tool buttons to make it possible to hide them
 * add tiled inference support for ScuNET
 * add branch support for extension installation
 * change linux installation script to insall into current directory rather than /home/username
 * sort textual inversion embeddings by name (case insensitive)
 * allow styles.csv to be symlinked or mounted in docker
 * remove the "do not add watermark to images" option
 * make selected tab configurable with UI config
 * extra networks UI in now fixed height and scrollable
 * add disable_tls_verify arg for use with self-signed certs

### Extensions:
 * Add reload callback
 * add is_hr_pass field for processing

### Bug Fixes:
 * fix broken batch image processing on 'Extras/Batch Process' tab
 * add "None" option to extra networks dropdowns
 * fix FileExistsError for CLIP Interrogator
 * fix /sdapi/v1/txt2img endpoint not working on Linux #9319
 * fix disappearing live previews and progressbar during slow tasks
 * fix fullscreen image view not working properly in some cases
 * prevent alwayson_scripts args param resizing script_arg list when they are inserted in it
 * fix prompt schedule for second order samplers
 * fix image mask/composite for weird resolutions #9628
 * use correct images for previews when using AND (see #9491)
 * one broken image in img2img batch won't stop all processing
 * fix image orientation bug in train/preprocess
 * fix Ngrok recreating tunnels every reload
 * fix --realesrgan-models-path and --ldsr-models-path not working
 * fix --skip-install not working
 * outpainting Mk2 & Poorman should use the SAMPLE file format to save images, not GRID file format
 * do not fail all Loras if some have failed to load when making a picture

## 1.0.0
  * everything
