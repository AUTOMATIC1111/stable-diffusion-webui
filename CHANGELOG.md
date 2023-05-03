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
