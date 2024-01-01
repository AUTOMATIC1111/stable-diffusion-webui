import gradio as gr

from modules import localization, ui_components, shared_items, shared, interrogate, shared_gradio_themes
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401
from modules.shared_cmd_options import cmd_opts
from modules.options import options_section, OptionInfo, OptionHTML, categories

options_templates = {}
hide_dirs = shared.hide_dirs

restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
    "outdir_init_images"
}

categories.register_category("saving", "Saving images")
categories.register_category("sd", "Stable Diffusion")
categories.register_category("ui", "User Interface")
categories.register_category("system", "System")
categories.register_category("postprocessing", "Postprocessing")
categories.register_category("training", "Training")

options_templates.update(options_section(('saving-images', "Saving images/grids", "saving"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('png', 'File format for images'),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),
    "save_images_replace_action": OptionInfo("Replace", "Saving the image to an existing file", gr.Radio, {"choices": ["Replace", "Add number suffix"], **hide_dirs}),
    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('png', 'File format for grids'),
    "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
    "grid_prevent_empty_spots": OptionInfo(False, "Prevent empty spots in grid (when set to autodetect)"),
    "grid_zip_filename_pattern": OptionInfo("", "Archive filename pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    "font": OptionInfo("", "Font for image grids that have text"),
    "grid_text_active_color": OptionInfo("#000000", "Text color for image grids", ui_components.FormColorPicker, {}),
    "grid_text_inactive_color": OptionInfo("#999999", "Inactive text color for image grids", ui_components.FormColorPicker, {}),
    "grid_background_color": OptionInfo("#ffffff", "Background color for image grids", ui_components.FormColorPicker, {}),

    "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration."),
    "save_images_before_highres_fix": OptionInfo(False, "Save a copy of image before applying highres fix."),
    "save_images_before_color_correction": OptionInfo(False, "Save a copy of image before applying color correction to img2img results"),
    "save_mask": OptionInfo(False, "For inpainting, save a copy of the greyscale mask"),
    "save_mask_composite": OptionInfo(False, "For inpainting, save a masked composite"),
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
    "export_for_4chan": OptionInfo(True, "Save copy of large images as JPG").info("if the file size is above the limit, or either width or height are above the limit"),
    "img_downscale_threshold": OptionInfo(4.0, "File size limit for the above option, MB", gr.Number),
    "target_side_length": OptionInfo(4000, "Width/height limit for the above option, in pixels", gr.Number),
    "img_max_size_mp": OptionInfo(200, "Maximum image size", gr.Number).info("in megapixels"),

    "use_original_name_batch": OptionInfo(True, "Use original name for output filename during batch process in extras tab"),
    "use_upscaler_name_as_suffix": OptionInfo(False, "Use upscaler name as filename suffix in the extras tab"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "save_init_img": OptionInfo(False, "Save init images when using img2img"),

    "temp_dir":  OptionInfo("", "Directory for temporary images; leave empty for default"),
    "clean_temp_dir_at_start": OptionInfo(False, "Cleanup non-default temporary directory when starting webui"),

    "save_incomplete_images": OptionInfo(False, "Save incomplete images").info("save images that has been interrupted in mid-generation; even if not saved, they will still show up in webui output."),

    "notification_audio": OptionInfo(True, "Play notification sound after image generation").info("notification.mp3 should be present in the root directory").needs_reload_ui(),
    "notification_volume": OptionInfo(100, "Notification sound volume", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}).info("in %"),
}))

options_templates.update(options_section(('saving-paths', "Paths for saving", "saving"), {
    "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output directory for images from extras tab', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids', component_args=hide_dirs),
    "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button", component_args=hide_dirs),
    "outdir_init_images": OptionInfo("outputs/init-images", "Directory for saving init images when using img2img", component_args=hide_dirs),
}))

options_templates.update(options_section(('saving-to-dirs', "Saving to a directory", "saving"), {
    "save_to_dirs": OptionInfo(True, "Save images to a subdirectory"),
    "grid_save_to_dirs": OptionInfo(True, "Save grids to a subdirectory"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "When using \"Save\" button, save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs).link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
}))

options_templates.update(options_section(('upscaling', "Upscaling", "postprocessing"), {
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}).info("0 = no tiling"),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap for ESRGAN upscalers.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}).info("Low values = visible seam"),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI.", gr.CheckboxGroup, lambda: {"choices": shared_items.realesrgan_models_names()}),
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in shared.sd_upscalers]}),
}))

options_templates.update(options_section(('face-restoration', "Face restoration", "postprocessing"), {
    "face_restoration": OptionInfo(False, "Restore faces", infotext='Face restoration').info("will use a third-party model on generation result to reconstruct faces"),
    "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in shared.face_restorers]}),
    "code_former_weight": OptionInfo(0.5, "CodeFormer weight", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}).info("0 = maximum effect; 1 = minimum effect"),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
}))

options_templates.update(options_section(('system', "System", "system"), {
    "auto_launch_browser": OptionInfo("Local", "Automatically open webui in browser on startup", gr.Radio, lambda: {"choices": ["Disable", "Local", "Remote"]}),
    "enable_console_prompts": OptionInfo(shared.cmd_opts.enable_console_prompts, "Print prompts to console when generating with txt2img and img2img."),
    "show_warnings": OptionInfo(False, "Show warnings in console.").needs_reload_ui(),
    "show_gradio_deprecation_warnings": OptionInfo(True, "Show gradio deprecation warnings in console.").needs_reload_ui(),
    "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}).info("0 = disable"),
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
    "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
    "print_hypernet_extra": OptionInfo(False, "Print extra hypernetwork information to console."),
    "list_hidden_files": OptionInfo(True, "Load models/files in hidden directories").info("directory is hidden if its name starts with \".\""),
    "disable_mmap_load_safetensors": OptionInfo(False, "Disable memmapping for loading .safetensors files.").info("fixes very slow loading speed in some cases"),
    "hide_ldm_prints": OptionInfo(True, "Prevent Stability-AI's ldm/sgm modules from printing noise to console."),
    "dump_stacks_on_signal": OptionInfo(False, "Print stack traces before exiting the program with ctrl+c."),
}))

options_templates.update(options_section(('API', "API", "system"), {
    "api_enable_requests": OptionInfo(True, "Allow http:// and https:// URLs for input images in API", restrict_api=True),
    "api_forbid_local_requests": OptionInfo(True, "Forbid URLs to local resources", restrict_api=True),
    "api_useragent": OptionInfo("", "User agent for requests", restrict_api=True),
}))

options_templates.update(options_section(('training', "Training", "training"), {
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training if possible. Saves VRAM."),
    "pin_memory": OptionInfo(False, "Turn on pin_memory for DataLoader. Makes training slightly faster but can increase memory usage."),
    "save_optimizer_state": OptionInfo(False, "Saves Optimizer state as separate *.optim file. Training of embedding or HN can be resumed with the matching optim file."),
    "save_training_settings_to_txt": OptionInfo(True, "Save textual inversion and hypernet settings to a text file whenever training starts."),
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex"),
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string"),
    "training_image_repeats_per_epoch": OptionInfo(1, "Number of repeats for a single input image per epoch; used only for displaying epoch number", gr.Number, {"precision": 0}),
    "training_write_csv_every": OptionInfo(500, "Save an csv containing the loss to log directory every N steps, 0 to disable"),
    "training_xattention_optimizations": OptionInfo(False, "Use cross attention optimizations while training"),
    "training_enable_tensorboard": OptionInfo(False, "Enable tensorboard logging."),
    "training_tensorboard_save_images": OptionInfo(False, "Save generated images within tensorboard."),
    "training_tensorboard_flush_every": OptionInfo(120, "How often, in seconds, to flush the pending tensorboard events and summaries to disk."),
}))

options_templates.update(options_section(('sd', "Stable Diffusion", "sd"), {
    "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)}, refresh=shared_items.refresh_checkpoints, infotext='Model hash'),
    "sd_checkpoints_limit": OptionInfo(1, "Maximum number of checkpoints loaded at the same time", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
    "sd_checkpoints_keep_in_cpu": OptionInfo(True, "Only keep one model on device").info("will keep models other than the currently used one in RAM rather than VRAM"),
    "sd_checkpoint_cache": OptionInfo(0, "Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}).info("obsolete; set to 0 and use the two settings above instead"),
    "sd_unet": OptionInfo("Automatic", "SD Unet", gr.Dropdown, lambda: {"choices": shared_items.sd_unet_items()}, refresh=shared_items.refresh_unet_list).info("choose Unet model: Automatic = use one with same filename as checkpoint; None = use Unet from checkpoint"),
    "enable_quantization": OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds").needs_reload_ui(),
    "enable_emphasis": OptionInfo(True, "Enable emphasis").info("use (text) to make model pay more attention to text and [text] to make it pay less attention"),
    "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
    "comma_padding_backtrack": OptionInfo(20, "Prompt word wrap length limit", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1}).info("in tokens - for texts shorter than specified, if they don't fit into 75 token limit, move them to the next 75 token chunk"),
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}, infotext="Clip skip").link("wiki", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#clip-skip").info("ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer"),
    "upcast_attn": OptionInfo(False, "Upcast cross attention layer to float32"),
    "randn_source": OptionInfo("GPU", "Random number generator source.", gr.Radio, {"choices": ["GPU", "CPU", "NV"]}, infotext="RNG").info("changes seeds drastically; use CPU to produce the same picture across different videocard vendors; use NV to produce same picture as on NVidia videocards"),
    "tiling": OptionInfo(False, "Tiling", infotext='Tiling').info("produce a tileable picture"),
    "hires_fix_refiner_pass": OptionInfo("second pass", "Hires fix: which pass to enable refiner for", gr.Radio, {"choices": ["first pass", "second pass", "both passes"]}, infotext="Hires refiner"),
}))

options_templates.update(options_section(('sdxl', "Stable Diffusion XL", "sd"), {
    "sdxl_crop_top": OptionInfo(0, "crop top coordinate"),
    "sdxl_crop_left": OptionInfo(0, "crop left coordinate"),
    "sdxl_refiner_low_aesthetic_score": OptionInfo(2.5, "SDXL low aesthetic score", gr.Number).info("used for refiner model negative prompt"),
    "sdxl_refiner_high_aesthetic_score": OptionInfo(6.0, "SDXL high aesthetic score", gr.Number).info("used for refiner model prompt"),
}))

options_templates.update(options_section(('vae', "VAE", "sd"), {
    "sd_vae_explanation": OptionHTML("""
<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>
image into latent space representation and back. Latent space representation is what stable diffusion is working on during sampling
(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting image after the sampling is finished.
For img2img, VAE is used to process user's input image before the sampling, and to create an image after sampling.
"""),
    "sd_vae_checkpoint_cache": OptionInfo(0, "VAE Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae": OptionInfo("Automatic", "SD VAE", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list, infotext='VAE').info("choose VAE model: Automatic = use one with same filename as checkpoint; None = use VAE from checkpoint"),
    "sd_vae_overrides_per_model_preferences": OptionInfo(True, "Selected VAE overrides per-model preferences").info("you can set per-model VAE either by editing user metadata for checkpoints, or by making the VAE have same name as checkpoint"),
    "auto_vae_precision": OptionInfo(True, "Automatically revert VAE to 32-bit floats").info("triggers when a tensor with NaNs is produced in VAE; disabling the option in this case will result in a black square image"),
    "sd_vae_encode_method": OptionInfo("Full", "VAE type for encode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext='VAE Encoder').info("method to encode image to latent (use in img2img, hires-fix or inpaint mask)"),
    "sd_vae_decode_method": OptionInfo("Full", "VAE type for decode", gr.Radio, {"choices": ["Full", "TAESD"]}, infotext='VAE Decoder').info("method to decode latent to image"),
}))

options_templates.update(options_section(('img2img', "img2img", "sd"), {
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Conditional mask weight'),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for img2img", gr.Slider, {"minimum": 0.0, "maximum": 1.5, "step": 0.001}, infotext='Noise multiplier'),
    "img2img_extra_noise": OptionInfo(0.0, "Extra noise multiplier for img2img and hires fix", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Extra noise').info("0 = disabled (default); should be lower than denoising strength"),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
    "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies.").info("normally you'd do less with less denoising"),
    "img2img_background_color": OptionInfo("#ffffff", "With img2img, fill transparent parts of the input image with this color.", ui_components.FormColorPicker, {}),
    "img2img_editor_height": OptionInfo(720, "Height of the image editor", gr.Slider, {"minimum": 80, "maximum": 1600, "step": 1}).info("in pixels").needs_reload_ui(),
    "img2img_sketch_default_brush_color": OptionInfo("#ffffff", "Sketch initial brush color", ui_components.FormColorPicker, {}).info("default brush color of img2img sketch").needs_reload_ui(),
    "img2img_inpaint_mask_brush_color": OptionInfo("#ffffff", "Inpaint mask brush color", ui_components.FormColorPicker,  {}).info("brush color of inpaint mask").needs_reload_ui(),
    "img2img_inpaint_sketch_default_brush_color": OptionInfo("#ffffff", "Inpaint sketch initial brush color", ui_components.FormColorPicker, {}).info("default brush color of img2img inpaint sketch").needs_reload_ui(),
    "return_mask": OptionInfo(False, "For inpainting, include the greyscale mask in results for web"),
    "return_mask_composite": OptionInfo(False, "For inpainting, include masked composite in results for web"),
    "img2img_batch_show_results_limit": OptionInfo(32, "Show the first N batch img2img results in UI", gr.Slider, {"minimum": -1, "maximum": 1000, "step": 1}).info('0: disable, -1: show all images. Too many images can cause lag'),
}))

options_templates.update(options_section(('optimizations', "Optimizations", "sd"), {
    "cross_attention_optimization": OptionInfo("Automatic", "Cross attention optimization", gr.Dropdown, lambda: {"choices": shared_items.cross_attention_optimizations()}),
    "s_min_uncond": OptionInfo(0.0, "Negative Guidance minimum sigma", gr.Slider, {"minimum": 0.0, "maximum": 15.0, "step": 0.01}).link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9177").info("skip negative prompt for some steps when the image is almost ready; 0=disable, higher=faster"),
    "token_merging_ratio": OptionInfo(0.0, "Token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}, infotext='Token merging ratio').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9256").info("0=disable, higher=faster"),
    "token_merging_ratio_img2img": OptionInfo(0.0, "Token merging ratio for img2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}).info("only applies if non-zero and overrides above"),
    "token_merging_ratio_hr": OptionInfo(0.0, "Token merging ratio for high-res pass", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}, infotext='Token merging ratio hr').info("only applies if non-zero and overrides above"),
    "pad_cond_uncond": OptionInfo(False, "Pad prompt/negative prompt to be same length", infotext='Pad conds').info("improves performance when prompt and negative prompt have different lengths; changes seeds"),
    "persistent_cond_cache": OptionInfo(True, "Persistent cond cache").info("do not recalculate conds from prompts if prompts have not changed since previous calculation"),
    "batch_cond_uncond": OptionInfo(True, "Batch cond/uncond").info("do both conditional and unconditional denoising in one batch; uses a bit more VRAM during sampling, but improves speed; previously this was controlled by --always-batch-cond-uncond comandline argument"),
}))

options_templates.update(options_section(('compatibility', "Compatibility", "sd"), {
    "use_old_emphasis_implementation": OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
    "use_old_karras_scheduler_sigmas": OptionInfo(False, "Use old karras scheduler sigmas (0.1 to 10)."),
    "no_dpmpp_sde_batch_determinism": OptionInfo(False, "Do not make DPM++ SDE deterministic across different batch sizes."),
    "use_old_hires_fix_width_height": OptionInfo(False, "For hires fix, use width/height sliders to set final resolution rather than first pass (disables Upscale by, Resize width/height to)."),
    "dont_fix_second_order_samplers_schedule": OptionInfo(False, "Do not fix prompt schedule for second order samplers."),
    "hires_fix_use_firstpass_conds": OptionInfo(False, "For hires fix, calculate conds of second pass using extra networks of first pass."),
    "use_old_scheduling": OptionInfo(False, "Use old prompt editing timelines.", infotext="Old prompt editing timelines").info("For [red:green:N]; old: If N < 1, it's a fraction of steps (and hires fix uses range from 0 to 1), if N >= 1, it's an absolute number of steps; new: If N has a decimal point in it, it's a fraction of steps (and hires fix uses range from 1 to 2), othewrwise it's an absolute number of steps"),
}))

options_templates.update(options_section(('interrogate', "Interrogate"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Keep models in VRAM"),
    "interrogate_return_ranks": OptionInfo(False, "Include ranks of model tags matches in results.").info("booru only"),
    "interrogate_clip_num_beams": OptionInfo(1, "BLIP: num_beams", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(24, "BLIP: minimum description length", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(48, "BLIP: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file").info("0 = No limit"),
    "interrogate_clip_skip_categories": OptionInfo([], "CLIP: skip inquire categories", gr.CheckboxGroup, lambda: {"choices": interrogate.category_types()}, refresh=interrogate.category_types),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "deepbooru: score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(True, "deepbooru: sort tags alphabetically").info("if not: sort by score"),
    "deepbooru_use_spaces": OptionInfo(True, "deepbooru: use spaces in tags").info("if not: use underscores"),
    "deepbooru_escape": OptionInfo(True, "deepbooru: escape (\\) brackets").info("so they are used as literal brackets and not for emphasis"),
    "deepbooru_filter_tags": OptionInfo("", "deepbooru: filter out those tags").info("separate by comma"),
}))

options_templates.update(options_section(('extra_networks', "Extra Networks", "sd"), {
    "extra_networks_show_hidden_directories": OptionInfo(True, "Show hidden directories").info("directory is hidden if its name starts with \".\"."),
    "extra_networks_dir_button_function": OptionInfo(False, "Add a '/' to the beginning of directory buttons").info("Buttons will display the contents of the selected directory without acting as a search filter."),
    "extra_networks_hidden_models": OptionInfo("When searched", "Show cards for models in hidden directories", gr.Radio, {"choices": ["Always", "When searched", "Never"]}).info('"When searched" option will only show the item when the search string has 4 characters or more'),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Default multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
    "extra_networks_card_width": OptionInfo(0, "Card width for Extra Networks").info("in pixels"),
    "extra_networks_card_height": OptionInfo(0, "Card height for Extra Networks").info("in pixels"),
    "extra_networks_card_text_scale": OptionInfo(1.0, "Card text scale", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}).info("1 = original size"),
    "extra_networks_card_show_desc": OptionInfo(True, "Show description on card"),
    "extra_networks_card_order_field": OptionInfo("Path", "Default order field for Extra Networks cards", gr.Dropdown, {"choices": ['Path', 'Name', 'Date Created', 'Date Modified']}).needs_reload_ui(),
    "extra_networks_card_order": OptionInfo("Ascending", "Default order for Extra Networks cards", gr.Dropdown, {"choices": ['Ascending', 'Descending']}).needs_reload_ui(),
    "extra_networks_add_text_separator": OptionInfo(" ", "Extra networks separator").info("extra text to add before <...> when adding extra network to prompt"),
    "ui_extra_networks_tab_reorder": OptionInfo("", "Extra networks tab order").needs_reload_ui(),
    "textual_inversion_print_at_load": OptionInfo(False, "Print a list of Textual Inversion embeddings when loading model"),
    "textual_inversion_add_hashes_to_infotext": OptionInfo(True, "Add Textual Inversion hashes to infotext"),
    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, lambda: {"choices": ["None", *shared.hypernetworks]}, refresh=shared_items.reload_hypernetworks),
}))

options_templates.update(options_section(('ui_prompt_editing', "Prompt editing", "ui"), {
    "keyedit_precision_attention": OptionInfo(0.1, "Precision for (attention:1.1) when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_precision_extra": OptionInfo(0.05, "Precision for <extra networks:0.9> when editing the prompt with Ctrl+up/down", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_delimiters": OptionInfo(r".,\/!?%^*;:{}=`~() ", "Word delimiters when editing the prompt with Ctrl+up/down"),
    "keyedit_delimiters_whitespace": OptionInfo(["Tab", "Carriage Return", "Line Feed"], "Ctrl+up/down whitespace delimiters", gr.CheckboxGroup, lambda: {"choices": ["Tab", "Carriage Return", "Line Feed"]}),
    "keyedit_move": OptionInfo(True, "Alt+left/right moves prompt elements"),
    "disable_token_counters": OptionInfo(False, "Disable prompt token counters").needs_reload_ui(),
}))

options_templates.update(options_section(('ui_gallery', "Gallery", "ui"), {
    "return_grid": OptionInfo(True, "Show grid in gallery"),
    "do_not_show_images": OptionInfo(False, "Do not show any images in gallery"),
    "js_modal_lightbox": OptionInfo(True, "Full page image viewer: enable"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Full page image viewer: show images zoomed in by default"),
    "js_modal_lightbox_gamepad": OptionInfo(False, "Full page image viewer: navigate with gamepad"),
    "js_modal_lightbox_gamepad_repeat": OptionInfo(250, "Full page image viewer: gamepad repeat period").info("in milliseconds"),
    "gallery_height": OptionInfo("", "Gallery height", gr.Textbox).info("can be any valid CSS value, for example 768px or 20em").needs_reload_ui(),
}))

options_templates.update(options_section(('ui_alternatives', "UI alternatives", "ui"), {
    "compact_prompt_box": OptionInfo(False, "Compact prompt layout").info("puts prompt and negative prompt inside the Generate tab, leaving more vertical space for the image on the right").needs_reload_ui(),
    "samplers_in_dropdown": OptionInfo(True, "Use dropdown for sampler selection instead of radio group").needs_reload_ui(),
    "dimensions_and_batch_together": OptionInfo(True, "Show Width/Height and Batch sliders in same row").needs_reload_ui(),
    "sd_checkpoint_dropdown_use_short": OptionInfo(False, "Checkpoint dropdown: use filenames without paths").info("models in subdirectories like photo/sd15.ckpt will be listed as just sd15.ckpt"),
    "hires_fix_show_sampler": OptionInfo(False, "Hires fix: show hires checkpoint and sampler selection").needs_reload_ui(),
    "hires_fix_show_prompts": OptionInfo(False, "Hires fix: show hires prompt and negative prompt").needs_reload_ui(),
    "txt2img_settings_accordion": OptionInfo(False, "Settings in txt2img hidden under Accordion").needs_reload_ui(),
    "img2img_settings_accordion": OptionInfo(False, "Settings in img2img hidden under Accordion").needs_reload_ui(),
}))

options_templates.update(options_section(('ui', "User interface", "ui"), {
    "localization": OptionInfo("None", "Localization", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)).needs_reload_ui(),
    "quicksettings_list": OptionInfo(["sd_model_checkpoint"], "Quicksettings list", ui_components.DropdownMulti, lambda: {"choices": list(shared.opts.data_labels.keys())}).js("info", "settingsHintsShowQuicksettings").info("setting entries that appear at the top of page rather than in settings tab").needs_reload_ui(),
    "ui_tab_order": OptionInfo([], "UI tab order", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
    "hidden_tabs": OptionInfo([], "Hidden UI tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared.tab_names)}).needs_reload_ui(),
    "ui_reorder_list": OptionInfo([], "UI item order for txt2img/img2img tabs", ui_components.DropdownMulti, lambda: {"choices": list(shared_items.ui_reorder_categories())}).info("selected items appear first").needs_reload_ui(),
    "gradio_theme": OptionInfo("Default", "Gradio theme", ui_components.DropdownEditable, lambda: {"choices": ["Default"] + shared_gradio_themes.gradio_hf_hub_themes}).info("you can also manually enter any of themes from the <a href='https://huggingface.co/spaces/gradio/theme-gallery'>gallery</a>.").needs_reload_ui(),
    "gradio_themes_cache": OptionInfo(True, "Cache gradio themes locally").info("disable to update the selected Gradio theme"),
    "show_progress_in_title": OptionInfo(True, "Show generation progress in window title."),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
}))


options_templates.update(options_section(('infotext', "Infotext", "ui"), {
    "infotext_explanation": OptionHTML("""
Infotext is what this software calls the text that contains generation parameters and can be used to generate the same picture again.
It is displayed in UI below the image. To use infotext, paste it into the prompt and click the ↙️ paste button.
"""),
    "enable_pnginfo": OptionInfo(True, "Write infotext to metadata of the generated image"),
    "save_txt": OptionInfo(False, "Create a text file with infotext next to every generated image"),

    "add_model_name_to_info": OptionInfo(True, "Add model name to infotext"),
    "add_model_hash_to_info": OptionInfo(True, "Add model hash to infotext"),
    "add_vae_name_to_info": OptionInfo(True, "Add VAE name to infotext"),
    "add_vae_hash_to_info": OptionInfo(True, "Add VAE hash to infotext"),
    "add_user_name_to_info": OptionInfo(False, "Add user name to infotext when authenticated"),
    "add_version_to_infotext": OptionInfo(True, "Add program version to infotext"),
    "disable_weights_auto_swap": OptionInfo(True, "Disregard checkpoint information from pasted infotext").info("when reading generation parameters from text into UI"),
    "infotext_skip_pasting": OptionInfo([], "Disregard fields from pasted infotext", ui_components.DropdownMulti, lambda: {"choices": shared_items.get_infotext_names()}),
    "infotext_styles": OptionInfo("Apply if any", "Infer styles from prompts of pasted infotext", gr.Radio, {"choices": ["Ignore", "Apply", "Discard", "Apply if any"]}).info("when reading generation parameters from text into UI)").html("""<ul style='margin-left: 1.5em'>
<li>Ignore: keep prompt and styles dropdown as it is.</li>
<li>Apply: remove style text from prompt, always replace styles dropdown value with found styles (even if none are found).</li>
<li>Discard: remove style text from prompt, keep styles dropdown as it is.</li>
<li>Apply if any: remove style text from prompt; if any styles are found in prompt, put them into styles dropdown, otherwise keep it as it is.</li>
</ul>"""),

}))

options_templates.update(options_section(('ui', "Live previews", "ui"), {
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    "live_previews_enable": OptionInfo(True, "Show live previews of the created image"),
    "live_previews_image_format": OptionInfo("png", "Live preview file format", gr.Radio, {"choices": ["jpeg", "png", "webp"]}),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    "show_progress_every_n_steps": OptionInfo(10, "Live preview display period", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}).info("in sampling steps - show new live preview image every N sampling steps; -1 = only show after completion of batch"),
    "show_progress_type": OptionInfo("Approx NN", "Live preview method", gr.Radio, {"choices": ["Full", "Approx NN", "Approx cheap", "TAESD"]}).info("Full = slow but pretty; Approx NN and TAESD = fast but low quality; Approx cheap = super fast but terrible otherwise"),
    "live_preview_allow_lowvram_full": OptionInfo(False, "Allow Full live preview method with lowvram/medvram").info("If not, Approx NN will be used instead; Full live preview method is very detrimental to speed if lowvram/medvram optimizations are enabled"),
    "live_preview_content": OptionInfo("Prompt", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"]}),
    "live_preview_refresh_period": OptionInfo(1000, "Progressbar and preview update period").info("in milliseconds"),
    "live_preview_fast_interrupt": OptionInfo(False, "Return image with chosen live preview method on interrupt").info("makes interrupts faster"),
    "js_live_preview_in_modal_lightbox": OptionInfo(False, "Show Live preview in full page image viewer"),
}))

options_templates.update(options_section(('sampler-params', "Sampler parameters", "sd"), {
    "hide_samplers": OptionInfo([], "Hide samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in shared_items.list_samplers()]}).needs_reload_ui(),
    "eta_ddim": OptionInfo(0.0, "Eta for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Eta DDIM').info("noise multiplier; higher = more unpredictable results"),
    "eta_ancestral": OptionInfo(1.0, "Eta for k-diffusion samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}, infotext='Eta').info("noise multiplier; currently only applies to ancestral samplers (i.e. Euler a) and SDE samplers"),
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM discretize", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 100.0, "step": 0.01}, infotext='Sigma churn').info('amount of stochasticity; only applies to Euler, Heun, and DPM2'),
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 10.0, "step": 0.01}, infotext='Sigma tmin').info('enable stochasticity; start value of the sigma range; only applies to Euler, Heun, and DPM2'),
    's_tmax':  OptionInfo(0.0, "sigma tmax",  gr.Slider, {"minimum": 0.0, "maximum": 999.0, "step": 0.01}, infotext='Sigma tmax').info("0 = inf; end value of the sigma range; only applies to Euler, Heun, and DPM2"),
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.1, "step": 0.001}, infotext='Sigma noise').info('amount of additional noise to counteract loss of detail during sampling'),
    'k_sched_type':  OptionInfo("Automatic", "Scheduler type", gr.Dropdown, {"choices": ["Automatic", "karras", "exponential", "polyexponential"]}, infotext='Schedule type').info("lets you override the noise schedule for k-diffusion samplers; choosing Automatic disables the three parameters below"),
    'sigma_min': OptionInfo(0.0, "sigma min", gr.Number, infotext='Schedule min sigma').info("0 = default (~0.03); minimum noise strength for k-diffusion noise scheduler"),
    'sigma_max': OptionInfo(0.0, "sigma max", gr.Number, infotext='Schedule max sigma').info("0 = default (~14.6); maximum noise strength for k-diffusion noise scheduler"),
    'rho':  OptionInfo(0.0, "rho", gr.Number, infotext='Schedule rho').info("0 = default (7 for karras, 1 for polyexponential); higher values result in a steeper noise schedule (decreases faster)"),
    'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}, infotext='ENSD').info("ENSD; does not improve anything, just produces different results for ancestral samplers - only useful for reproducing images"),
    'always_discard_next_to_last_sigma': OptionInfo(False, "Always discard next-to-last sigma", infotext='Discard penultimate sigma').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044"),
    'sgm_noise_multiplier': OptionInfo(False, "SGM noise multiplier", infotext='SGM noise multplier').link("PR", "https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12818").info("Match initial noise to official SDXL implementation - only useful for reproducing images"),
    'uni_pc_variant': OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}, infotext='UniPC variant'),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}, infotext='UniPC skip type'),
    'uni_pc_order': OptionInfo(3, "UniPC order", gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}, infotext='UniPC order').info("must be < sampling steps"),
    'uni_pc_lower_order_final': OptionInfo(True, "UniPC lower order final", infotext='UniPC lower order final'),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing", "postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Enable postprocessing operations in txt2img and img2img tabs", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'upscaling_max_images_in_cache': OptionInfo(5, "Maximum number of images in upscaling cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    'postprocessing_existing_caption_action': OptionInfo("Ignore", "Action for existing captions", gr.Radio, {"choices": ["Ignore", "Keep", "Prepend", "Append"]}).info("when generating captions using postprocessing; Ignore = use generated; Keep = use original; Prepend/Append = combine both"),
}))

options_templates.update(options_section((None, "Hidden options"), {
    "disabled_extensions": OptionInfo([], "Disable these extensions"),
    "disable_all_extensions": OptionInfo("none", "Disable all extensions (preserves the list of disabled extensions)", gr.Radio, {"choices": ["none", "extra", "all"]}),
    "restore_config_state_file": OptionInfo("", "Config state file to restore from, under 'config-states/' folder"),
    "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint"),
}))
