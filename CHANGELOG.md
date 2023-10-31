## 1.6.0

### Features:
 * refiner support [#12371](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12371)
 * add NV option for Random number generator source setting, which allows to generate same pictures on CPU/AMD/Mac as on NVidia videocards
 * add style editor dialog
 * hires fix: add an option to use a different checkpoint for second pass ([#12181](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12181))
 * option to keep multiple loaded models in memory ([#12227](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12227))
 * new samplers: Restart, DPM++ 2M SDE Exponential, DPM++ 2M SDE Heun, DPM++ 2M SDE Heun Karras, DPM++ 2M SDE Heun Exponential, DPM++ 3M SDE, DPM++ 3M SDE Karras, DPM++ 3M SDE Exponential ([#12300](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12300), [#12519](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12519), [#12542](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12542))
 * rework DDIM, PLMS, UniPC to use CFG denoiser same as in k-diffusion samplers:
   * makes all of them work with img2img
   * makes prompt composition posssible (AND)
   * makes them available for SDXL
 * always show extra networks tabs in the UI ([#11808](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/11808))
 * use less RAM when creating models ([#11958](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/11958), [#12599](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12599))
 * textual inversion inference support for SDXL
 * extra networks UI: show metadata for SD checkpoints
 * checkpoint merger: add metadata support 
 * prompt editing and attention: add support for whitespace after the number ([ red : green : 0.5 ]) (seed breaking change) ([#12177](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12177))
 * VAE: allow selecting own VAE for each checkpoint (in user metadata editor)
 * VAE: add selected VAE to infotext
 * options in main UI: add own separate setting for txt2img and img2img, correctly read values from pasted infotext, add setting for column count ([#12551](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12551))
 * add resize handle to txt2img and img2img tabs, allowing to change the amount of horizontable space given to generation parameters and resulting image gallery ([#12687](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12687), [#12723](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12723))
 * change default behavior for batching cond/uncond -- now it's on by default, and is disabled by an UI setting (Optimizatios -> Batch cond/uncond) - if you are on lowvram/medvram and are getting OOM exceptions, you will need to enable it
 * show current position in queue and make it so that requests are processed in the order of arrival ([#12707](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12707))
 * add `--medvram-sdxl` flag that only enables `--medvram` for SDXL models
 * prompt editing timeline has separate range for first pass and hires-fix pass (seed breaking change) ([#12457](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12457))

### Minor:
 * img2img batch: RAM savings, VRAM savings, .tif, .tiff in img2img batch ([#12120](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12120), [#12514](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12514), [#12515](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12515))
 * postprocessing/extras: RAM savings ([#12479](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12479))
 * XYZ: in the axis labels, remove pathnames from model filenames
 * XYZ: support hires sampler ([#12298](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12298))
 * XYZ: new option: use text inputs instead of dropdowns ([#12491](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12491))
 * add gradio version warning
 * sort list of VAE checkpoints ([#12297](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12297))
 * use transparent white for mask in inpainting, along with an option to select the color ([#12326](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12326))
 * move some settings to their own section: img2img, VAE
 * add checkbox to show/hide dirs for extra networks
 * Add TAESD(or more) options for all the VAE encode/decode operation ([#12311](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12311))
 * gradio theme cache, new gradio themes, along with explanation that the user can input his own values ([#12346](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12346), [#12355](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12355))
 * sampler fixes/tweaks: s_tmax, s_churn, s_noise, s_tmax ([#12354](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12354), [#12356](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12356), [#12357](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12357), [#12358](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12358), [#12375](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12375), [#12521](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12521))
 * update README.md with correct instructions for Linux installation ([#12352](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12352))
 * option to not save incomplete images, on by default ([#12338](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12338))
 * enable cond cache by default
 * git autofix for repos that are corrupted ([#12230](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12230))
 * allow to open images in new browser tab by middle mouse button ([#12379](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12379))
 * automatically open webui in browser when running "locally" ([#12254](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12254))
 * put commonly used samplers on top, make DPM++ 2M Karras the default choice
 * zoom and pan: option to auto-expand a wide image, improved integration ([#12413](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12413), [#12727](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12727))
 * option to cache Lora networks in memory
 * rework hires fix UI to use accordion
 * face restoration and tiling moved to settings - use "Options in main UI" setting if you want them back
 * change quicksettings items to have variable width
 * Lora: add Norm module, add support for bias ([#12503](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12503))
 * Lora: output warnings in UI rather than fail for unfitting loras; switch to logging for error output in console
 * support search and display of hashes for all extra network items ([#12510](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12510))
 * add extra noise param for img2img operations ([#12564](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12564))
 * support for Lora with bias ([#12584](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12584))
 * make interrupt quicker ([#12634](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12634))
 * configurable gallery height ([#12648](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12648))
 * make results column sticky ([#12645](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12645))
 * more hash filename patterns ([#12639](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12639))
 * make image viewer actually fit the whole page ([#12635](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12635))
 * make progress bar work independently from live preview display which results in it being updated a lot more often
 * forbid Full live preview method for medvram and add a setting to undo the forbidding
 * make it possible to localize tooltips and placeholders
 * add option to align with sgm repo's sampling implementation ([#12818](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818))
 * Restore faces and Tiling generation parameters have been moved to settings out of main UI
   * if you want to put them back into main UI, use `Options in main UI` setting on the UI page.

### Extensions and API:
 * gradio 3.41.2
 * also bump versions for packages: transformers, GitPython, accelerate, scikit-image, timm, tomesd
 * support tooltip kwarg for gradio elements: gr.Textbox(label='hello', tooltip='world')
 * properly clear the total console progressbar when using txt2img and img2img from API
 * add cmd_arg --disable-extra-extensions and --disable-all-extensions ([#12294](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12294))
 * shared.py and webui.py split into many files
 * add --loglevel commandline argument for logging
 * add a custom UI element that combines accordion and checkbox
 * avoid importing gradio in tests because it spams warnings
 * put infotext label for setting into OptionInfo definition rather than in a separate list
 * make `StableDiffusionProcessingImg2Img.mask_blur` a property, make more inline with PIL `GaussianBlur` ([#12470](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12470))
 * option to make scripts UI without gr.Group
 * add a way for scripts to register a callback for before/after just a single component's creation
 * use dataclass for StableDiffusionProcessing
 * store patches for Lora in a specialized module instead of inside torch
 * support http/https URLs in API ([#12663](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12663), [#12698](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12698))
 * add extra noise callback ([#12616](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12616))
 * dump current stack traces when exiting with SIGINT
 * add type annotations for extra fields of shared.sd_model

### Bug Fixes:
 * Don't crash if out of local storage quota for javascriot localStorage
 * XYZ plot do not fail if an exception occurs
 * fix missing TI hash in infotext if generation uses both negative and positive TI ([#12269](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12269))
 * localization fixes ([#12307](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12307))
 * fix sdxl model invalid configuration after the hijack
 * correctly toggle extras checkbox for infotext paste ([#12304](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12304))
 * open raw sysinfo link in new page ([#12318](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12318))
 * prompt parser: Account for empty field in alternating words syntax ([#12319](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12319))
 * add tab and carriage return to invalid filename chars ([#12327](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12327))
 * fix api only Lora not working ([#12387](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12387))
 * fix options in main UI misbehaving when there's just one element
 * make it possible to use a sampler from infotext even if it's hidden in the dropdown
 * fix styles missing from the prompt in infotext when making a grid of batch of multiplie images
 * prevent bogus progress output in console when calculating hires fix dimensions
 * fix --use-textbox-seed
 * fix broken `Lora/Networks: use old method` option ([#12466](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12466))
 * properly return `None` for VAE hash when using `--no-hashing` ([#12463](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12463))
 * MPS/macOS fixes and optimizations ([#12526](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12526))
 * add second_order to samplers that mistakenly didn't have it
 * when refreshing cards in extra networks UI, do not discard user's custom resolution
 * fix processing error that happens if batch_size is not a multiple of how many prompts/negative prompts there are ([#12509](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12509))
 * fix inpaint upload for alpha masks ([#12588](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12588))
 * fix exception when image sizes are not integers ([#12586](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12586))
 * fix incorrect TAESD Latent scale ([#12596](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12596))
 * auto add data-dir to gradio-allowed-path ([#12603](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12603))
 * fix exception if extensuions dir is missing ([#12607](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12607))
 * fix issues with api model-refresh and vae-refresh ([#12638](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12638))
 * fix img2img background color for transparent images option not being used ([#12633](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12633))
 * attempt to resolve NaN issue with unstable VAEs in fp32 mk2 ([#12630](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12630))
 * implement missing undo hijack for SDXL
 * fix xyz swap axes ([#12684](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12684))
 * fix errors in backup/restore tab if any of config files are broken ([#12689](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12689))
 * fix SD VAE switch error after model reuse ([#12685](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12685))
 * fix trying to create images too large for the chosen format ([#12667](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12667))
 * create Gradio temp directory if necessary ([#12717](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12717))
 * prevent possible cache loss if exiting as it's being written by using an atomic operation to replace the cache with the new version
 * set devices.dtype_unet correctly
 * run RealESRGAN on GPU for non-CUDA devices ([#12737](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12737))
 * prevent extra network buttons being obscured by description for very small card sizes ([#12745](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12745))
 * fix error that causes some extra networks to be disabled if both <lora:> and <lyco:> are present in the prompt
 * fix defaults settings page breaking when any of main UI tabs are hidden
 * fix incorrect save/display of new values in Defaults page in settings
 * fix for Reload UI function: if you reload UI on one tab, other opened tabs will no longer stop working
 * fix an error that prevents VAE being reloaded after an option change if a VAE near the checkpoint exists ([#12797](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12737))
 * hide broken image crop tool ([#12792](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12737))
 * don't show hidden samplers in dropdown for XYZ script ([#12780](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12737))
 * fix style editing dialog breaking if it's opened in both img2img and txt2img tabs
 * fix a bug allowing users to bypass gradio and API authentication (reported by vysecurity) 
 * fix notification not playing when built-in webui tab is inactive ([#12834](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12834))
 * honor `--skip-install` for extension installers ([#12832](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12832))
 * don't print blank stdout in extension installers ([#12833](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12832), [#12855](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12855))
 * do not change quicksettings dropdown option when value returned is `None` ([#12854](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12854))
 * get progressbar to display correctly in extensions tab


## 1.5.2

### Bug Fixes:
 * fix memory leak when generation fails
 * update doggettx cross attention optimization to not use an unreasonable amount of memory in some edge cases -- suggestion by MorkTheOrk


## 1.5.1

### Minor:
 * support parsing text encoder blocks in some new LoRAs
 * delete scale checker script due to user demand

### Extensions and API:
 * add postprocess_batch_list script callback

### Bug Fixes:
 * fix TI training for SD1
 * fix reload altclip model error
 * prepend the pythonpath instead of overriding it
 * fix typo in SD_WEBUI_RESTARTING
 * if txt2img/img2img raises an exception, finally call state.end()
 * fix composable diffusion weight parsing
 * restyle Startup profile for black users
 * fix webui not launching with --nowebui
 * catch exception for non git extensions
 * fix some options missing from /sdapi/v1/options
 * fix for extension update status always saying "unknown"
 * fix display of extra network cards that have `<>` in the name
 * update lora extension to work with python 3.8


## 1.5.0

### Features:
 * SD XL support
 * user metadata system for custom networks
 * extended Lora metadata editor: set activation text, default weight, view tags, training info
 * Lora extension rework to include other types of networks (all that were previously handled by LyCORIS extension)
 * show github stars for extenstions
 * img2img batch mode can read extra stuff from png info
 * img2img batch works with subdirectories
 * hotkeys to move prompt elements: alt+left/right
 * restyle time taken/VRAM display
 * add textual inversion hashes to infotext
 * optimization: cache git extension repo information
 * move generate button next to the generated picture for mobile clients
 * hide cards for networks of incompatible Stable Diffusion version in Lora extra networks interface
 * skip installing packages with pip if they all are already installed - startup speedup of about 2 seconds

### Minor:
 * checkbox to check/uncheck all extensions in the Installed tab
 * add gradio user to infotext and to filename patterns
 * allow gif for extra network previews
 * add options to change colors in grid
 * use natural sort for items in extra networks
 * Mac: use empty_cache() from torch 2 to clear VRAM
 * added automatic support for installing the right libraries for Navi3 (AMD)
 * add option SWIN_torch_compile to accelerate SwinIR upscale
 * suppress printing TI embedding info at start to console by default
 * speedup extra networks listing
 * added `[none]` filename token.
 * removed thumbs extra networks view mode (use settings tab to change width/height/scale to get thumbs)
 * add always_discard_next_to_last_sigma option to XYZ plot
 * automatically switch to 32-bit float VAE if the generated picture has NaNs without the need for `--no-half-vae` commandline flag.
 
### Extensions and API:
 * api endpoints: /sdapi/v1/server-kill, /sdapi/v1/server-restart, /sdapi/v1/server-stop
 * allow Script to have custom metaclass
 * add model exists status check /sdapi/v1/options
 * rename --add-stop-route to --api-server-stop
 * add `before_hr` script callback
 * add callback `after_extra_networks_activate`
 * disable rich exception output in console for API by default, use WEBUI_RICH_EXCEPTIONS env var to enable
 * return http 404 when thumb file not found
 * allow replacing extensions index with environment variable
 
### Bug Fixes:
 * fix for catch errors when retrieving extension index #11290
 * fix very slow loading speed of .safetensors files when reading from network drives
 * API cache cleanup
 * fix UnicodeEncodeError when writing to file CLIP Interrogator batch mode
 * fix warning of 'has_mps' deprecated from PyTorch
 * fix problem with extra network saving images as previews losing generation info
 * fix throwing exception when trying to resize image with I;16 mode
 * fix for #11534: canvas zoom and pan extension hijacking shortcut keys
 * fixed launch script to be runnable from any directory
 * don't add "Seed Resize: -1x-1" to API image metadata
 * correctly remove end parenthesis with ctrl+up/down
 * fixing --subpath on newer gradio version
 * fix: check fill size none zero when resize  (fixes #11425)
 * use submit and blur for quick settings textbox
 * save img2img batch with images.save_image()
 * prevent running preload.py for disabled extensions
 * fix: previously, model name was added together with directory name to infotext and to [model_name] filename pattern; directory name is now not included


## 1.4.1

### Bug Fixes:
 * add queue lock for refresh-checkpoints

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
