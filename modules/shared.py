import argparse
import datetime
import json
import os
import sys
import time

from PIL import Image
import gradio as gr
import tqdm

import modules.interrogate
import modules.memmon
import modules.styles
import modules.devices as devices
from modules import localization, script_loading, errors, ui_components, shared_items, cmd_args
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir

demo = None

parser = cmd_args.parser

script_loading.preload_extensions(extensions_dir, parser)
script_loading.preload_extensions(extensions_builtin_dir, parser)

if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
    cmd_opts = parser.parse_args()
else:
    cmd_opts, _ = parser.parse_known_args()


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
}

ui_reorder_categories = [
    "inpaint",
    "sampler",
    "checkboxes",
    "hires_fix",
    "dimensions",
    "cfg",
    "seed",
    "batch",
    "override_settings",
    "scripts",
]

cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or cmd_opts.server_name) and not cmd_opts.enable_insecure_extension_access

devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = \
    (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])

device = devices.device
weight_load_location = None if cmd_opts.lowram else "cpu"

batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram
xformers_available = False
config_filename = cmd_opts.ui_settings_file

os.makedirs(cmd_opts.hypernetwork_dir, exist_ok=True)
hypernetworks = {}
loaded_hypernetworks = []


def reload_hypernetworks():
    from modules.hypernetworks import hypernetwork
    global hypernetworks

    hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)


class State:
    skipped = False
    interrupted = False
    job = ""
    job_no = 0
    job_count = 0
    processing_has_refined_job_count = False
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    id_live_preview = 0
    textinfo = None
    time_start = None
    need_restart = False
    server_start = None

    def skip(self):
        self.skipped = True

    def interrupt(self):
        self.interrupted = True

    def nextjob(self):
        if opts.live_previews_enable and opts.show_progress_every_n_steps == -1:
            self.do_set_current_image()

        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "job": self.job,
            "job_count": self.job_count,
            "job_timestamp": self.job_timestamp,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }

        return obj

    def begin(self):
        self.sampling_step = 0
        self.job_count = -1
        self.processing_has_refined_job_count = False
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_latent = None
        self.current_image = None
        self.current_image_sampling_step = 0
        self.id_live_preview = 0
        self.skipped = False
        self.interrupted = False
        self.textinfo = None
        self.time_start = time.time()

        devices.torch_gc()

    def end(self):
        self.job = ""
        self.job_count = 0

        devices.torch_gc()

    def set_current_image(self):
        """sets self.current_image from self.current_latent if enough sampling steps have been made after the last call to this"""
        if not parallel_processing_allowed:
            return

        if self.sampling_step - self.current_image_sampling_step >= opts.show_progress_every_n_steps and opts.live_previews_enable and opts.show_progress_every_n_steps != -1:
            self.do_set_current_image()

    def do_set_current_image(self):
        if self.current_latent is None:
            return

        import modules.sd_samplers
        if opts.show_progress_grid:
            self.assign_current_image(modules.sd_samplers.samples_to_image_grid(self.current_latent))
        else:
            self.assign_current_image(modules.sd_samplers.sample_to_image(self.current_latent))

        self.current_image_sampling_step = self.sampling_step

    def assign_current_image(self, image):
        self.current_image = image
        self.id_live_preview += 1


state = State()
state.server_start = time.time()

styles_filename = cmd_opts.styles_file
prompt_styles = modules.styles.StyleDatabase(styles_filename)

interrogator = modules.interrogate.InterrogateModels("interrogate")

face_restorers = []

class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh


def options_section(section_identifier, options_dict):
    for k, v in options_dict.items():
        v.section = section_identifier

    return options_dict


def list_checkpoint_tiles():
    import modules.sd_models
    return modules.sd_models.checkpoint_tiles()


def refresh_checkpoints():
    import modules.sd_models
    return modules.sd_models.list_models()


def list_samplers():
    import modules.sd_samplers
    return modules.sd_samplers.all_samplers


hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
tab_names = []

options_templates = {}

options_templates.update(options_section(('saving-images', "Saving images/grids"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('png', 'File format for images'),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern", component_args=hide_dirs),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),

    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('png', 'File format for grids'),
    "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
    "grid_prevent_empty_spots": OptionInfo(False, "Prevent empty spots in grid (when set to autodetect)"),
    "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),

    "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
    "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters."),
    "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration."),
    "save_images_before_highres_fix": OptionInfo(False, "Save a copy of image before applying highres fix."),
    "save_images_before_color_correction": OptionInfo(False, "Save a copy of image before applying color correction to img2img results"),
    "save_mask": OptionInfo(False, "For inpainting, save a copy of the greyscale mask"),
    "save_mask_composite": OptionInfo(False, "For inpainting, save a masked composite"),
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
    "export_for_4chan": OptionInfo(True, "If the saved image file size is above the limit, or its either width or height are above the limit, save a downscaled copy as JPG"),
    "img_downscale_threshold": OptionInfo(4.0, "File size limit for the above option, MB", gr.Number),
    "target_side_length": OptionInfo(4000, "Width/height limit for the above option, in pixels", gr.Number),
    "img_max_size_mp": OptionInfo(200, "Maximum image size, in megapixels", gr.Number),

    "use_original_name_batch": OptionInfo(True, "Use original name for output filename during batch process in extras tab"),
    "use_upscaler_name_as_suffix": OptionInfo(False, "Use upscaler name as filename suffix in the extras tab"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "do_not_add_watermark": OptionInfo(False, "Do not add watermark to images"),

    "temp_dir":  OptionInfo("", "Directory for temporary images; leave empty for default"),
    "clean_temp_dir_at_start": OptionInfo(False, "Cleanup non-default temporary directory when starting webui"),

}))

options_templates.update(options_section(('saving-paths', "Paths for saving"), {
    "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo("outputs/extras-images", 'Output directory for images from extras tab', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids', component_args=hide_dirs),
    "outdir_save": OptionInfo("log/images", "Directory for saving images using the Save button", component_args=hide_dirs),
}))

options_templates.update(options_section(('saving-to-dirs', "Saving to a directory"), {
    "save_to_dirs": OptionInfo(True, "Save images to a subdirectory"),
    "grid_save_to_dirs": OptionInfo(True, "Save grids to a subdirectory"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "When using \"Save\" button, save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
}))

options_templates.update(options_section(('upscaling', "Upscaling"), {
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for ESRGAN upscalers. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI. (Requires restart)", gr.CheckboxGroup, lambda: {"choices": shared_items.realesrgan_models_names()}),
    "upscaler_for_img2img": OptionInfo(None, "Upscaler for img2img", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
}))

options_templates.update(options_section(('face-restoration', "Face restoration"), {
    "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.5, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
}))

options_templates.update(options_section(('system', "System"), {
    "show_warnings": OptionInfo(False, "Show warnings in console."),
    "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation. Set to 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}),
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
    "multiple_tqdm": OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
    "print_hypernet_extra": OptionInfo(False, "Print extra hypernetwork information to console."),
}))

options_templates.update(options_section(('training', "Training"), {
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

options_templates.update(options_section(('sd', "Stable Diffusion"), {
    "sd_model_checkpoint": OptionInfo(None, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_checkpoint_cache": OptionInfo(0, "Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae_checkpoint_cache": OptionInfo(0, "VAE Checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae": OptionInfo("Automatic", "SD VAE", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list),
    "sd_vae_as_default": OptionInfo(True, "Ignore selected VAE for stable diffusion checkpoints that have their own .vae.pt next to them"),
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for img2img", gr.Slider, {"minimum": 0.5, "maximum": 1.5, "step": 0.01}),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
    "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies (normally you'd do less with less denoising)."),
    "img2img_background_color": OptionInfo("#ffffff", "With img2img, fill image's transparent parts with this color.", ui_components.FormColorPicker, {}),
    "enable_quantization": OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply."),
    "enable_emphasis": OptionInfo(True, "Emphasis: use (text) to make model pay more attention to text and [text] to make it pay less attention"),
    "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
    "comma_padding_backtrack": OptionInfo(20, "Increase coherency by padding from the last comma within n tokens when using more than 75 tokens", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1 }),
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}),
    "upcast_attn": OptionInfo(False, "Upcast cross attention layer to float32"),
}))

options_templates.update(options_section(('compatibility', "Compatibility"), {
    "use_old_emphasis_implementation": OptionInfo(False, "Use old emphasis implementation. Can be useful to reproduce old seeds."),
    "use_old_karras_scheduler_sigmas": OptionInfo(False, "Use old karras scheduler sigmas (0.1 to 10)."),
    "no_dpmpp_sde_batch_determinism": OptionInfo(False, "Do not make DPM++ SDE deterministic across different batch sizes."),
    "use_old_hires_fix_width_height": OptionInfo(False, "For hires fix, use width/height sliders to set final resolution rather than first pass (disables Upscale by, Resize width/height to)."),
}))

options_templates.update(options_section(('interrogate', "Interrogate Options"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Interrogate: keep models in VRAM"),
    "interrogate_return_ranks": OptionInfo(False, "Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators)."),
    "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(48, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(1500, "CLIP: maximum number of lines in text file (0 = No limit)"),
    "interrogate_clip_skip_categories": OptionInfo([], "CLIP: skip inquire categories", gr.CheckboxGroup, lambda: {"choices": modules.interrogate.category_types()}, refresh=modules.interrogate.category_types),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "Interrogate: deepbooru score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(True, "Interrogate: deepbooru sort alphabetically"),
    "deepbooru_use_spaces": OptionInfo(False, "use spaces for tags in deepbooru"),
    "deepbooru_escape": OptionInfo(True, "escape (\\) brackets in deepbooru (so they are used as literal brackets and not for emphasis)"),
    "deepbooru_filter_tags": OptionInfo("", "filter out those tags from deepbooru output (separated by comma)"),
}))

options_templates.update(options_section(('extra_networks', "Extra Networks"), {
    "extra_networks_default_view": OptionInfo("cards", "Default view for Extra Networks", gr.Dropdown, {"choices": ["cards", "thumbs"]}),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "extra_networks_card_width": OptionInfo(0, "Card width for Extra Networks (px)"),
    "extra_networks_card_height": OptionInfo(0, "Card height for Extra Networks (px)"),
    "extra_networks_add_text_separator": OptionInfo(" ", "Extra text to add before <...> when adding extra network to prompt"),
    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, lambda: {"choices": [""] + [x for x in hypernetworks.keys()]}, refresh=reload_hypernetworks),
}))

options_templates.update(options_section(('ui', "User interface"), {
    "return_grid": OptionInfo(True, "Show grid in results for web"),
    "return_mask": OptionInfo(False, "For inpainting, include the greyscale mask in results for web"),
    "return_mask_composite": OptionInfo(False, "For inpainting, include masked composite in results for web"),
    "do_not_show_images": OptionInfo(False, "Do not show any images in results for web"),
    "add_model_hash_to_info": OptionInfo(True, "Add model hash to generation information"),
    "add_model_name_to_info": OptionInfo(True, "Add model name to generation information"),
    "disable_weights_auto_swap": OptionInfo(True, "When reading generation parameters from text into UI (from PNG info or pasted text), do not change the selected model/checkpoint."),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
    "font": OptionInfo("", "Font for image grids that have text"),
    "js_modal_lightbox": OptionInfo(True, "Enable full page image viewer"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Show images zoomed in by default in full page image viewer"),
    "show_progress_in_title": OptionInfo(True, "Show generation progress in window title."),
    "samplers_in_dropdown": OptionInfo(True, "Use dropdown for sampler selection instead of radio group"),
    "dimensions_and_batch_together": OptionInfo(True, "Show Width/Height and Batch sliders in same row"),
    "keyedit_precision_attention": OptionInfo(0.1, "Ctrl+up/down precision when editing (attention:1.1)", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_precision_extra": OptionInfo(0.05, "Ctrl+up/down precision when editing <extra networks:0.9>", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "quicksettings": OptionInfo("sd_model_checkpoint", "Quicksettings list"),
    "hidden_tabs": OptionInfo([], "Hidden UI tabs (requires restart)", ui_components.DropdownMulti, lambda: {"choices": [x for x in tab_names]}),
    "ui_reorder": OptionInfo(", ".join(ui_reorder_categories), "txt2img/img2img UI item order"),
    "ui_extra_networks_tab_reorder": OptionInfo("", "Extra networks tab order"),
    "localization": OptionInfo("None", "Localization (requires restart)", gr.Dropdown, lambda: {"choices": ["None"] + list(localization.localizations.keys())}, refresh=lambda: localization.list_localizations(cmd_opts.localizations_dir)),
}))

options_templates.update(options_section(('ui', "Live previews"), {
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    "live_previews_enable": OptionInfo(True, "Show live previews of the created image"),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    "show_progress_every_n_steps": OptionInfo(10, "Show new live preview image every N sampling steps. Set to -1 to show after completion of batch.", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "show_progress_type": OptionInfo("Approx NN", "Image creation progress preview mode", gr.Radio, {"choices": ["Full", "Approx NN", "Approx cheap"]}),
    "live_preview_content": OptionInfo("Prompt", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"]}),
    "live_preview_refresh_period": OptionInfo(1000, "Progressbar/preview update period, in milliseconds")
}))

options_templates.update(options_section(('sampler-params', "Sampler parameters"), {
    "hide_samplers": OptionInfo([], "Hide samplers in user interface (requires restart)", gr.CheckboxGroup, lambda: {"choices": [x.name for x in list_samplers()]}),
    "eta_ddim": OptionInfo(0.0, "eta (noise multiplier) for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "eta_ancestral": OptionInfo(1.0, "eta (noise multiplier) for ancestral samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM discretize", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    'eta_noise_seed_delta': OptionInfo(0, "Eta noise seed delta", gr.Number, {"precision": 0}),
    'always_discard_next_to_last_sigma': OptionInfo(False, "Always discard next-to-last sigma"),
    'uni_pc_variant': OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}),
    'uni_pc_order': OptionInfo(3, "UniPC order (must be < sampling steps)", gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}),
    'uni_pc_lower_order_final': OptionInfo(True, "UniPC lower order final"),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Enable postprocessing operations in txt2img and img2img tabs", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'upscaling_max_images_in_cache': OptionInfo(5, "Maximum number of images in upscaling cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
}))

options_templates.update(options_section((None, "Hidden options"), {
    "disabled_extensions": OptionInfo([], "Disable these extensions"),
    "disable_all_extensions": OptionInfo("none", "Disable all extensions (preserves the list of disabled extensions)", gr.Radio, {"choices": ["none", "extra", "all"]}),
    "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint"),
}))

options_templates.update()


class Options:
    data = None
    data_labels = options_templates
    typemap = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data or key in self.data_labels:
                assert not cmd_opts.freeze_settings, "changing settings is disabled"

                info = opts.data_labels.get(key, None)
                comp_args = info.component_args if info else None
                if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                if cmd_opts.hide_ui_dir_config and key in restricted_opts:
                    raise RuntimeError(f"not possible to set {key} because it is restricted")

                self.data[key] = value
                return

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.data_labels:
            return self.data_labels[item].default

        return super(Options, self).__getattribute__(item)

    def set(self, key, value):
        """sets an option and calls its onchange callback, returning True if the option changed and False otherwise"""

        oldval = self.data.get(key, None)
        if oldval == value:
            return False

        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        if self.data_labels[key].onchange is not None:
            try:
                self.data_labels[key].onchange()
            except Exception as e:
                errors.display(e, f"changing setting {key} to {value}")
                setattr(self, key, oldval)
                return False

        return True

    def get_default(self, key):
        """returns the default value for the key"""

        data_label = self.data_labels.get(key)
        if data_label is None:
            return None

        return data_label.default

    def save(self, filename):
        assert not cmd_opts.freeze_settings, "saving settings is disabled"

        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.data, file, indent=4)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.data = json.load(file)

        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func

        if call:
            func()

    def dumpjson(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        return json.dumps(d)

    def add_option(self, key, info):
        self.data_labels[key] = info

    def reorder(self):
        """reorder settings so that all items related to section always go together"""

        section_ids = {}
        settings_items = self.data_labels.items()
        for k, item in settings_items:
            if item.section not in section_ids:
                section_ids[item.section] = len(section_ids)

        self.data_labels = {k: v for k, v in sorted(settings_items, key=lambda x: section_ids[x[1].section])}

    def cast_value(self, key, value):
        """casts an arbitrary to the same type as this setting's value with key
        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)
        """

        if value is None:
            return None

        default_value = self.data_labels[key].default
        if default_value is None:
            default_value = getattr(self, key, None)
        if default_value is None:
            return None

        expected_type = type(default_value)
        if expected_type == bool and value == "False":
            value = False
        else:
            value = expected_type(value)

        return value



opts = Options()
if os.path.exists(config_filename):
    opts.load(config_filename)

settings_components = None
"""assinged from ui.py, a mapping on setting anmes to gradio components repsponsible for those settings"""

latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}

sd_upscalers = []

sd_model = None

clip_model = None

progress_print_out = sys.stdout


class TotalTQDM:
    def __init__(self):
        self._tqdm = None

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc="Total progress",
            total=state.job_count * state.sampling_steps,
            position=1,
            file=progress_print_out
        )

    def update(self):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()

    def updateTotal(self, new_total):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.total = new_total

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.refresh()
            self._tqdm.close()
            self._tqdm = None


total_tqdm = TotalTQDM()

mem_mon = modules.memmon.MemUsageMonitor("MemMon", device, opts)
mem_mon.start()


def listfiles(dirname):
    filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname), key=str.lower) if not x.startswith(".")]
    return [file for file in filenames if os.path.isfile(file)]


def html_path(filename):
    return os.path.join(script_path, "html", filename)


def html(filename):
    path = html_path(filename)

    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()

    return ""
