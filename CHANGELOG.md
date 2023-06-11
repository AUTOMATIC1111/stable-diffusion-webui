## 1.4.0

### Features:
 * zoom controls for inpainting
 * run basic torch calculation at startup in parallel to reduce the performance impact of first generation
 * option to pad prompt/neg prompt to be same length
 * remove taming_transformers dependency
 * custom k-diffusion scheduler settings
 * add an option to show selected settings in main txt2img/img2img UI
 * sysinfo tab in settings
 * infer styles from prompts when pasting params into the UI
 * an option to control the behavior of the above

### Minor:
 * bump Gradio to 3.32.0
 * bump xformers to 0.0.20
 * Add option to disable token counters
 * tooltip fixes & optimizations
 * make it possible to configure filename for the zip download
 * `[vae_filename]` pattern for filenames
 * Revert discarding penultimate sigma for DPM-Solver++(2M) SDE
 * change UI reorder setting to multiselect
 * read version info form CHANGELOG.md if git version info is not available
 * link footer API to Wiki when API is not active
 * persistent conds cache (opt-in optimization)
 
### Extensions:
 * After installing extensions, webui properly restarts the process rather than reloads the UI 
 * Added VAE listing to web API. Via: /sdapi/v1/sd-vae
 * custom unet support
 * Add onAfterUiUpdate callback
 * refactor EmbeddingDatabase.register_embedding() to allow unregistering
 * add before_process callback for scripts
 * add ability for alwayson scripts to specify section and let user reorder those sections
 
### Bug Fixes:
 * Fix dragging text to prompt
 * fix incorrect quoting for infotext values with colon in them
 * fix "hires. fix" prompt sharing same labels with txt2img_prompt
 * Fix s_min_uncond default type int
 * Fix for #10643 (Inpainting mask sometimes not working)
 * fix bad styling for thumbs view in extra networks #10639
 * fix for empty list of optimizations #10605
 * small fixes to prepare_tcmalloc for Debian/Ubuntu compatibility
 * fix --ui-debug-mode exit
 * patch GitPython to not use leaky persistent processes
 * fix duplicate Cross attention optimization after UI reload
 * torch.cuda.is_available() check for SdOptimizationXformers
 * fix hires fix using wrong conds in second pass if using Loras.
 * handle exception when parsing generation parameters from png info
 * fix upcast attention dtype error
 * forcing Torch Version to 1.13.1 for RX 5000 series GPUs
 * split mask blur into X and Y components, patch Outpainting MK2 accordingly
 * don't die when a LoRA is a broken symlink
 * allow activation of Generate Forever during generation


## 1.3.2

### Bug Fixes:
 * fix files served out of tmp directory even if they are saved to disk
 * fix postprocessing overwriting parameters

## 1.3.1

### Features:
 * revert default cross attention optimization to Doggettx

### Bug Fixes:
 * fix bug: LoRA don't apply on dropdown list sd_lora
 * fix png info always added even if setting is not enabled
 * fix some fields not applying in xyz plot
 * fix "hires. fix" prompt sharing same labels with txt2img_prompt
 * fix lora hashes not being added properly to infotex if there is only one lora
 * fix --use-cpu failing to work properly at startup
 * make --disable-opt-split-attention command line option work again

## 1.3.0

### Features:
 * add UI to edit defaults
 * token merging (via dbolya/tomesd)
 * settings tab rework: add a lot of additional explanations and links
 * load extensions' Git metadata in parallel to loading the main program to save a ton of time during startup
 * update extensions table: show branch, show date in separate column, and show version from tags if available
 * TAESD - another option for cheap live previews
 * allow choosing sampler and prompts for second pass of hires fix - hidden by default, enabled in settings
 * calculate hashes for Lora
 * add lora hashes to infotext
 * when pasting infotext, use infotext's lora hashes to find local loras for `<lora:xxx:1>` entries whose hashes match loras the user has
 * select cross attention optimization from UI

### Minor:
 * bump Gradio to 3.31.0
 * bump PyTorch to 2.0.1 for macOS and Linux AMD
 * allow setting defaults for elements in extensions' tabs
 * allow selecting file type for live previews
 * show "Loading..." for extra networks when displaying for the first time
 * suppress ENSD infotext for samplers that don't use it
 * clientside optimizations
 * add options to show/hide hidden files and dirs in extra networks, and to not list models/files in hidden directories
 * allow whitespace in styles.csv
 * add option to reorder tabs
 * move some functionality (swap resolution and set seed to -1) to client
 * option to specify editor height for img2img
 * button to copy image resolution into img2img width/height sliders
 * switch from pyngrok to ngrok-py
 * lazy-load images in extra networks UI
 * set "Navigate image viewer with gamepad" option to false by default, by request
 * change upscalers to download models into user-specified directory (from commandline args) rather than the default models/<...>
 * allow hiding buttons in ui-config.json

### Extensions:
 * add /sdapi/v1/script-info api
 * use Ruff to lint Python code
 * use ESlint to lint Javascript code
 * add/modify CFG callbacks for Self-Attention Guidance extension
 * add command and endpoint for graceful server stopping
 * add some locals (prompts/seeds/etc) from processing function into the Processing class as fields
 * rework quoting for infotext items that have commas in them to use JSON (should be backwards compatible except for cases where it didn't work previously)
 * add /sdapi/v1/refresh-loras api checkpoint post request
 * tests overhaul

### Bug Fixes:
 * fix an issue preventing the program from starting if the user specifies a bad Gradio theme
 * fix broken prompts from file script
 * fix symlink scanning for extra networks
 * fix --data-dir ignored when launching via webui-user.bat COMMANDLINE_ARGS
 * allow web UI to be ran fully offline
 * fix inability to run with --freeze-settings
 * fix inability to merge checkpoint without adding metadata
 * fix extra networks' save preview image not adding infotext for jpeg/webm
 * remove blinking effect from text in hires fix and scale resolution preview
 * make links to `http://<...>.git` extensions work in the extension tab
 * fix bug with webui hanging at startup due to hanging git process


## 1.2.1

### Features:
 * add an option to always refer to LoRA by filenames

### Bug Fixes:
 * never refer to LoRA by an alias if multiple LoRAs have same alias or the alias is called none
 * fix upscalers disappearing after the user reloads UI
 * allow bf16 in safe unpickler (resolves problems with loading some LoRAs)
 * allow web UI to be ran fully offline
 * fix localizations not working
 * fix error for LoRAs: `'LatentDiffusion' object has no attribute 'lora_layer_mapping'`

## 1.2.0

### Features:
 * do not wait for Stable Diffusion model to load at startup
 * add filename patterns: `[denoising]`
 * directory hiding for extra networks: dirs starting with `.` will hide their cards on extra network tabs unless specifically searched for
 * LoRA: for the `<...>` text in prompt, use name of LoRA that is in the metdata of the file, if present, instead of filename (both can be used to activate LoRA)
 * LoRA: read infotext params from kohya-ss's extension parameters if they are present and if his extension is not active
 * LoRA: fix some LoRAs not working (ones that have 3x3 convolution layer)
 * LoRA: add an option to use old method of applying LoRAs (producing same results as with kohya-ss)
 * add version to infotext, footer and console output when starting
 * add links to wiki for filename pattern settings
 * add extended info for quicksettings setting and use multiselect input instead of a text field

### Minor:
 * bump Gradio to 3.29.0
 * bump PyTorch to 2.0.1
 * `--subpath` option for gradio for use with reverse proxy
 * Linux/macOS: use existing virtualenv if already active (the VIRTUAL_ENV environment variable)
 * do not apply localizations if there are none (possible frontend optimization)
 * add extra `None` option for VAE in XYZ plot
 * print error to console when batch processing in img2img fails
 * create HTML for extra network pages only on demand
 * allow directories starting with `.` to still list their models for LoRA, checkpoints, etc
 * put infotext options into their own category in settings tab
 * do not show licenses page when user selects Show all pages in settings

### Extensions:
 * tooltip localization support
 * add API method to get LoRA models with prompt

### Bug Fixes:
 * re-add `/docs` endpoint
 * fix gamepad navigation
 * make the lightbox fullscreen image function properly
 * fix squished thumbnails in extras tab
 * keep "search" filter for extra networks when user refreshes the tab (previously it showed everthing after you refreshed)
 * fix webui showing the same image if you configure the generation to always save results into same file
 * fix bug with upscalers not working properly
 * fix MPS on PyTorch 2.0.1, Intel Macs
 * make it so that custom context menu from contextMenu.js only disappears after user's click, ignoring non-user click events
 * prevent Reload UI button/link from reloading the page when it's not yet ready
 * fix prompts from file script failing to read contents from a drag/drop file


## 1.1.1
### Bug Fixes:
 * fix an error that prevents running webui on PyTorch<2.0 without --disable-safe-unpickle

## 1.1.0
### Features:
 * switch to PyTorch 2.0.0 (except for AMD GPUs)
 * visual improvements to custom code scripts
 * add filename patterns: `[clip_skip]`, `[hasprompt<>]`, `[batch_number]`, `[generation_number]`
 * add support for saving init images in img2img, and record their hashes in infotext for reproducability
 * automatically select current word when adjusting weight with ctrl+up/down
 * add dropdowns for X/Y/Z plot
 * add setting: Stable Diffusion/Random number generator source: makes it possible to make images generated from a given manual seed consistent across different GPUs
 * support Gradio's theme API
 * use TCMalloc on Linux by default; possible fix for memory leaks
 * add optimization option to remove negative conditioning at low sigma values #9177
 * embed model merge metadata in .safetensors file
 * extension settings backup/restore feature #9169
 * add "resize by" and "resize to" tabs to img2img
 * add option "keep original size" to textual inversion images preprocess
 * image viewer scrolling via analog stick
 * button to restore the progress from session lost / tab reload

### Minor:
 * bump Gradio to 3.28.1
 * change "scale to" to sliders in Extras tab
 * add labels to tool buttons to make it possible to hide them
 * add tiled inference support for ScuNET
 * add branch support for extension installation
 * change Linux installation script to install into current directory rather than `/home/username`
 * sort textual inversion embeddings by name (case-insensitive)
 * allow styles.csv to be symlinked or mounted in docker
 * remove the "do not add watermark to images" option
 * make selected tab configurable with UI config
 * make the extra networks UI fixed height and scrollable
 * add `disable_tls_verify` arg for use with self-signed certs

### Extensions:
 * add reload callback
 * add `is_hr_pass` field for processing

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
 * fix `--realesrgan-models-path` and `--ldsr-models-path` not working
 * fix `--skip-install` not working
 * use SAMPLE file format in Outpainting Mk2 & Poorman
 * do not fail all LoRAs if some have failed to load when making a picture

## 1.0.0
  * everything
