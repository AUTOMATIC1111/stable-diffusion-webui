import os
import sys
import time
import json
import datetime
import gradio as gr
import tqdm
import modules.interrogate
import modules.memmon
import modules.styles
import modules.devices as devices
from modules import errors, ui_components, shared_items, cmd_args
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir # pylint: disable=W0611
import modules.paths_internal as paths
from setup import log as setup_log # pylint: disable=E0611

errors.install(gr)
demo: gr.Blocks = None
log = setup_log
parser = cmd_args.parser
url = 'https://github.com/vladmandic/automatic'
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

cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or cmd_opts.server_name) and not cmd_opts.enable_insecure
devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])
device = devices.device
is_device_dml = False
sd_upscalers = []
sd_model = None
clip_model = None


if device.type == 'privateuseone':
    import modules.dml
    is_device_dml = True


def reload_hypernetworks():
    from modules.hypernetworks import hypernetwork
    global hypernetworks # pylint: disable=W0603
    hypernetworks = hypernetwork.list_hypernetworks(opts.hypernetwork_dir)


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
        import modules.sd_samplers # pylint: disable=W0621
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
    for _k, v in options_dict.items():
        v.section = section_identifier
    return options_dict


def list_checkpoint_tiles():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.checkpoint_tiles()


def refresh_checkpoints():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.list_models()


def list_samplers():
    import modules.sd_samplers # pylint: disable=W0621
    return modules.sd_samplers.all_samplers

def list_themes():
    if not os.path.exists(os.path.join('javascript', 'themes.json')):
        refresh_themes()
    if os.path.exists(os.path.join('javascript', 'themes.json')):
        with open(os.path.join('javascript', 'themes.json'), mode='r', encoding='utf=8') as f:
            res = json.loads(f.read())
    else:
        res = []
    builtin = ["black-orange", "gradio/default", "gradio/base", "gradio/glass", "gradio/monochrome", "gradio/soft"]
    themes = builtin + [x['id'] for x in res if x['status'] == 'RUNNING' and 'test' not in x['id'].lower()]
    return themes


def refresh_themes():
    import requests
    try:
        req = requests.get('https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json', timeout=5)
        if req.status_code == 200:
            res = req.json()
            with open(os.path.join('javascript', 'themes.json'), mode='w', encoding='utf=8') as f:
                f.write(json.dumps(res))
        else:
            print('Error refreshing UI themes')
    except:
        print('Exception refreshing UI themes')

hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
tab_names = []
options_templates = {}
default_checkpoint = list_checkpoint_tiles()[0] if len(list_checkpoint_tiles()) > 0 else "model.ckpt"

options_templates.update(options_section(('sd', "Stable Diffusion"), {
    "sd_model_checkpoint": OptionInfo(default_checkpoint, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_checkpoint_cache": OptionInfo(0, "Model checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae_checkpoint_cache": OptionInfo(0, "VAE checkpoints to cache in RAM", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae": OptionInfo("Automatic", "Select VAE", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list),
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for img2img", gr.Slider, {"minimum": 0.5, "maximum": 1.5, "step": 0.01}),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
    "img2img_fix_steps": OptionInfo(False, "For image processing do exactly the amount of steps as specified."),
    "img2img_background_color": OptionInfo("#ffffff", "With img2img, fill image's transparent parts with this color.", ui_components.FormColorPicker, {}),
    "enable_quantization": OptionInfo(True, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds."),
    "comma_padding_backtrack": OptionInfo(20, "Increase coherency by padding from the last comma within n tokens when using more than 75 tokens", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1 }),
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1, "visible": False}),
    "upcast_attn": OptionInfo(False, "Upcast cross attention layer to float32"),
    "cross_attention_optimization": OptionInfo("Sub-quadratic" if is_device_dml else "Scaled-Dot-Product", "Cross-attention optimization method", gr.Radio, lambda: {"choices": shared_items.list_crossattention() }),
    "cross_attention_options": OptionInfo([], "Cross-attention advanced options", gr.CheckboxGroup, lambda: {"choices": ['xFormers enable flash Attention', 'SDP disable memory attention']}),
    "sub_quad_q_chunk_size": OptionInfo(512, "Sub-quadratic cross-attention query chunk size for the  layer optimization to use", gr.Slider, {"minimum": 16, "maximum": 8192, "step": 8}),
    "sub_quad_kv_chunk_size": OptionInfo(512, "Sub-quadratic cross-attentionkv chunk size for the sub-quadratic cross-attention layer optimization to use", gr.Slider, {"minimum": 0, "maximum": 8192, "step": 8}),
    "sub_quad_chunk_threshold": OptionInfo(80, "Sub-quadratic cross-attention percentage of VRAM chunking threshold", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),
    "always_batch_cond_uncond": OptionInfo(False, "Disables cond/uncond batching that is enabled to save memory with --medvram or --lowvram"),
}))

options_templates.update(options_section(('system-paths', "System Paths"), {
    "temp_dir":  OptionInfo("", "Directory for temporary images; leave empty for default"),
    "clean_temp_dir_at_start": OptionInfo(True, "Cleanup non-default temporary directory when starting webui"),
    "ckpt_dir": OptionInfo(os.path.join(paths.models_path, 'Stable-diffusion'), "Path to directory with stable diffusion checkpoints"),
    "vae_dir": OptionInfo(os.path.join(paths.models_path, 'VAE'), "Path to directory with VAE files"),
    "embeddings_dir": OptionInfo(os.path.join(paths.models_path, 'embeddings'), "Embeddings directory for textual inversion"),
    "embeddings_templates_dir": OptionInfo(os.path.join(paths.script_path, 'train/templates'), "Embeddings train templates directory"),
    "embeddings_train_log": OptionInfo(os.path.join(paths.script_path, 'train.csv'), "Embeddings train log file"),
    "hypernetwork_dir": OptionInfo(os.path.join(paths.models_path, 'hypernetworks'), "Hypernetwork directory"),
    "codeformer_models_path": OptionInfo(os.path.join(paths.models_path, 'Codeformer'), "Path to directory with codeformer model file(s)."),
    "gfpgan_models_path": OptionInfo(os.path.join(paths.models_path, 'GFPGAN'), "Path to directory with GFPGAN model file(s)"),
    "esrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'ESRGAN'), "Path to directory with ESRGAN model file(s)"),
    "bsrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'BSRGAN'), "Path to directory with BSRGAN model file(s)"),
    "realesrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'RealESRGAN'), "Path to directory with RealESRGAN model file(s)"),
    "scunet_models_path": OptionInfo(os.path.join(paths.models_path, 'ScuNET'), "Path to directory with ScuNET model file(s)"),
    "swinir_models_path": OptionInfo(os.path.join(paths.models_path, 'SwinIR'), "Path to directory with SwinIR model file(s)"),
    "ldsr_models_path": OptionInfo(os.path.join(paths.models_path, 'LDSR'), "Path to directory with LDSR model file(s)"),
    "clip_models_path": OptionInfo(os.path.join(paths.models_path, 'CLIP'), "Path to directory with CLIP model file(s)"),
    "lora_dir": OptionInfo(os.path.join(paths.models_path, 'Lora'), "Path to directory with Lora network(s)"),
    "lyco_dir": OptionInfo(os.path.join(paths.models_path, 'LyCORIS'), "Path to directory with LyCORIS network(s)"),
    "styles_dir": OptionInfo('styles.csv', "Path to user-defined styles file"),
    # "gfpgan_model": OptionInfo("", "GFPGAN model file name"),
}))

options_templates.update(options_section(('saving-images', "Image options"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('jpg', 'File format for images'),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern", component_args=hide_dirs),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),
    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('jpg', 'File format for grids'),
    "grid_extended_filename": OptionInfo(True, "Add extended info (seed, prompt) to filename when saving grid"),
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
    "grid_prevent_empty_spots": OptionInfo(True, "Prevent empty spots in grid (when set to autodetect)"),
    "n_rows": OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
    "save_txt": OptionInfo(False, "Create a text file next to every image with generation parameters."),
    "save_images_before_face_restoration": OptionInfo(True, "Save a copy of image before doing face restoration."),
    "save_images_before_highres_fix": OptionInfo(True, "Save a copy of image before applying highres fix."),
    "save_images_before_color_correction": OptionInfo(True, "Save a copy of image before applying color correction to img2img results"),
    "save_mask": OptionInfo(False, "For inpainting, save a copy of the greyscale mask"),
    "save_mask_composite": OptionInfo(False, "For inpainting, save a masked composite"),
    "jpeg_quality": OptionInfo(85, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
    "img_max_size_mp": OptionInfo(200, "Maximum image size, in megapixels", gr.Number),
    "use_original_name_batch": OptionInfo(True, "Use original name for output filename during batch process in extras tab"),
    "use_upscaler_name_as_suffix": OptionInfo(True, "Use upscaler name as filename suffix in the extras tab"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "save_to_dirs": OptionInfo(False, "Save images to a subdirectory"),
    "grid_save_to_dirs": OptionInfo(False, "Save grids to a subdirectory"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "When using \"Save\" button, save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1, **hide_dirs}),
}))

options_templates.update(options_section(('saving-paths', "Image Paths"), {
    "outdir_samples": OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo("outputs/text", 'Output directory for txt2img images', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo("outputs/image", 'Output directory for img2img images', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo("outputs/extras", 'Output directory for images from extras tab', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo("outputs/grids", 'Output directory for txt2img grids', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo("outputs/grids", 'Output directory for img2img grids', component_args=hide_dirs),
    "outdir_save": OptionInfo("outputs/save", "Directory for saving images using the Save button", component_args=hide_dirs),
}))

options_templates.update(options_section(('cuda', "CUDA Settings"), {
    "memmon_poll_rate": OptionInfo(2, "VRAM usage polls per second during generation. Set to 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}),
    "precision": OptionInfo("Autocast", "Precision type", gr.Radio, lambda: {"choices": ["Autocast", "Full"]}),
    "cuda_dtype": OptionInfo("FP32" if sys.platform == "darwin" else "FP16", "Device precision type", gr.Radio, lambda: {"choices": ["FP32", "FP16", "BF16"]}),
    "no_half": OptionInfo(True if is_device_dml else False, "Use full precision for model (--no-half)", None, None, lambda: print("Warning: Most of DirectML devices do not fully support half mode. Recommend to use full precision to model.") if is_device_dml else None),
    "no_half_vae": OptionInfo(True if is_device_dml else False, "Use full precision for VAE (--no-half-vae)"),
    "upcast_sampling": OptionInfo(True if sys.platform == "darwin" else False, "Enable upcast sampling. Usually produces similar results to --no-half with better performance while using less memory"),
    "disable_nan_check": OptionInfo(True, "Do not check if produced images/latent spaces have NaN values"),
    "rollback_vae": OptionInfo(False, "Attempt to roll back VAE when produced NaN values, requires NaN check (experimental)"),
    "opt_channelslast": OptionInfo(False, "Use channels last as torch memory format "),
    "cudnn_benchmark": OptionInfo(False, "Enable cuDNN benchmark feature"),
    "cuda_allow_tf32": OptionInfo(True, "Allow TF32 math ops"),
    "cuda_allow_tf16_reduced": OptionInfo(True, "Allow TF16 reduced precision math ops"),
    "cuda_compile": OptionInfo(False, "Enable model compile (experimental)"),
    "cuda_compile_mode": OptionInfo("none", "Model compile mode (experimental)", gr.Radio, lambda: {"choices": ['none', 'inductor', 'cudagraphs', 'aot_ts_nvfuser']}),
}))

options_templates.update(options_section(('upscaling', "Upscaling"), {
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers (0 = no tiling)", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap in pixels for ESRGAN upscalers", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Real-ESRGAN available models", gr.CheckboxGroup, lambda: {"choices": shared_items.realesrgan_models_names()}),
    "upscaler_for_img2img": OptionInfo("None", "Default upscaler for image resize operations", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
    "use_old_hires_fix_width_height": OptionInfo(False, "Hires fix uses width & height to set final resolution rather than first pass"),
    "dont_fix_second_order_samplers_schedule": OptionInfo(False, "Do not fix prompt schedule for second order samplers."),
}))

options_templates.update(options_section(('face-restoration', "Face restoration"), {
    "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.2, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
}))

options_templates.update(options_section(('training', "Training"), {
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training if possible. Saves VRAM."),
    "pin_memory": OptionInfo(True, "Turn on pin_memory for DataLoader. Makes training slightly faster but can increase memory usage."),
    "save_optimizer_state": OptionInfo(False, "Saves Optimizer state as separate *.optim file. Training of embedding or HN can be resumed with the matching optim file."),
    "save_training_settings_to_txt": OptionInfo(True, "Save textual inversion and hypernet settings to a text file whenever training starts."),
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex"),
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string"),
    "training_image_repeats_per_epoch": OptionInfo(1, "Number of repeats for a single input image per epoch; used only for displaying epoch number", gr.Number, {"precision": 0}),
    "training_write_csv_every": OptionInfo(0, "Save an csv containing the loss to log directory every N steps, 0 to disable"),
    "training_enable_tensorboard": OptionInfo(False, "Enable tensorboard logging."),
    "training_tensorboard_save_images": OptionInfo(False, "Save generated images within tensorboard."),
    "training_tensorboard_flush_every": OptionInfo(120, "How often, in seconds, to flush the pending tensorboard events and summaries to disk."),
}))

options_templates.update(options_section(('interrogate', "Interrogate Options"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Interrogate: keep models in VRAM"),
    "interrogate_return_ranks": OptionInfo(True, "Interrogate: include ranks of model tags matches in results (Has no effect on caption-based interrogators)."),
    "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(32, "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(192, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(2048, "CLIP: maximum number of lines in text file (0 = No limit)"),
    "interrogate_clip_skip_categories": OptionInfo(["artists", "movements", "flavors"], "CLIP: skip inquire categories", gr.CheckboxGroup, lambda: {"choices": modules.interrogate.category_types()}, refresh=modules.interrogate.category_types),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.65, "Interrogate: deepbooru score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(False, "Interrogate: deepbooru sort alphabetically"),
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
    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, lambda: {"choices": ["None"] + [x for x in hypernetworks.keys()]}, refresh=reload_hypernetworks),
}))

options_templates.update(options_section(('ui', "User interface"), {
    "gradio_theme": OptionInfo("black-orange", "UI theme", gr.Dropdown, lambda: {"choices": list_themes()}, refresh=refresh_themes),
    "return_grid": OptionInfo(True, "Show grid in results for web"),
    "return_mask": OptionInfo(False, "For inpainting, include the greyscale mask in results for web"),
    "return_mask_composite": OptionInfo(False, "For inpainting, include masked composite in results for web"),
    "disable_weights_auto_swap": OptionInfo(True, "Do not change the selected model when reading generation parameters."),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
    "font": OptionInfo("", "Font for image grids that have text"),
    "keyedit_precision_attention": OptionInfo(0.1, "Ctrl+up/down precision when editing (attention:1.1)", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_precision_extra": OptionInfo(0.05, "Ctrl+up/down precision when editing <extra networks:0.9>", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "quicksettings": OptionInfo("sd_model_checkpoint", "Quicksettings list"),
    "hidden_tabs": OptionInfo([], "Hidden UI tabs", ui_components.DropdownMulti, lambda: {"choices": [x for x in tab_names]}),
    "ui_reorder": OptionInfo(", ".join(ui_reorder_categories), "txt2img/img2img UI item order"),
    "ui_extra_networks_tab_reorder": OptionInfo("", "Extra networks tab order"),
}))

options_templates.update(options_section(('ui', "Live previews"), {
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    "live_previews_enable": OptionInfo(True, "Show live previews of the created image"),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    "show_progress_every_n_steps": OptionInfo(1, "Show new live preview image every N sampling steps. Set to -1 to show after completion of batch.", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "show_progress_type": OptionInfo("Approx NN", "Image creation progress preview mode", gr.Radio, {"choices": ["Full", "Approx NN", "Approx cheap"]}),
    "live_preview_content": OptionInfo("Combined", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"]}),
    "live_preview_refresh_period": OptionInfo(250, "Progressbar/preview update period, in milliseconds")
}))

options_templates.update(options_section(('sampler-params', "Sampler parameters"), {
    "show_samplers": OptionInfo(["Euler a", "UniPC", "DDIM", "DPM++ SDE", "DPM++ SDE", "DPM2 Karras", "DPM++ 2M Karras"], "Show samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in list_samplers()]}),
    "fallback_sampler": OptionInfo("Euler a", "Secondary sampler", gr.Dropdown, lambda: {"choices": ["None"] + [x.name for x in list_samplers()]}),
    "eta_ancestral": OptionInfo(1.0, "Noise multiplier for ancestral samplers (eta)", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "eta_ddim": OptionInfo(0.0, "Noise multiplier for DDIM (eta)", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "ddim_discretize": OptionInfo('uniform', "DDIM discretize img2img", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    'eta_noise_seed_delta': OptionInfo(0, "Noise seed delta (eta)", gr.Number, {"precision": 0}),
    'always_discard_next_to_last_sigma': OptionInfo(False, "Always discard next-to-last sigma"),
    'uni_pc_variant': OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}),
    'uni_pc_order': OptionInfo(3, "UniPC order (must be < sampling steps)", gr.Slider, {"minimum": 1, "maximum": 50, "step": 1}),
    'uni_pc_lower_order_final': OptionInfo(True, "UniPC lower order final"),
}))

options_templates.update(options_section(('token_merging', 'Token Merging'), {
    "token_merging": OptionInfo(False, "Enable redundant token merging via tomesd. This can provide significant speed and memory improvements.", gr.Checkbox),
    "token_merging_ratio": OptionInfo(0.5, "Merging Ratio", gr.Slider, {"minimum": 0, "maximum": 0.9, "step": 0.1}),
    "token_merging_hr_only": OptionInfo(True, "Apply only to high-res fix pass. Disabling can yield a ~20-35% speedup on contemporary resolutions.", gr.Checkbox),
    "token_merging_ratio_hr": OptionInfo(0.5, "Merging Ratio (high-res pass) - If 'Apply only to high-res' is enabled, this will always be the ratio used.", gr.Slider, {"minimum": 0, "maximum": 0.9, "step": 0.1}),
    "token_merging_random": OptionInfo(False, "Use random perturbations - Can improve outputs for certain samplers. For others, it may cause visual artifacting.", gr.Checkbox),
    "token_merging_merge_attention": OptionInfo(True, "Merge attention", gr.Checkbox),
    "token_merging_merge_cross_attention": OptionInfo(False, "Merge cross attention", gr.Checkbox),
    "token_merging_merge_mlp": OptionInfo(False, "Merge mlp", gr.Checkbox),
    "token_merging_maximum_down_sampling": OptionInfo(1, "Maximum down sampling", gr.Dropdown, lambda: {"choices": ["1", "2", "4", "8"]}),
    "token_merging_stride_x": OptionInfo(2, "Stride - X", gr.Slider, {"minimum": 2, "maximum": 8, "step": 2}),
    "token_merging_stride_y": OptionInfo(2, "Stride - Y", gr.Slider, {"minimum": 2, "maximum": 8, "step": 2})
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
                if cmd_opts.freeze_settings:
                    print(f'Settings are frozen: {key}')
                    return
                if cmd_opts.hide_ui_dir_config and key in restricted_opts:
                    print(f'Settings key is restricted: {key}')
                    return
                else:
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
                log.error(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            log.error(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.", file=sys.stderr)

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
batch_cond_uncond = opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram
xformers_available = False
config_filename = cmd_opts.ui_settings_file
os.makedirs(opts.hypernetwork_dir, exist_ok=True)
hypernetworks = {}
loaded_hypernetworks = []
if os.path.exists(config_filename):
    opts.load(config_filename)
cmd_opts = cmd_args.compatibility_args(opts, cmd_opts)
prompt_styles = modules.styles.StyleDatabase(opts.styles_dir)
settings_components = None
"""assinged from ui.py, a mapping on setting names to gradio components repsponsible for those settings"""
latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}
progress_print_out = sys.stdout
gradio_theme = gr.themes.Base()


def reload_gradio_theme(theme_name=None):
    global gradio_theme # pylint: disable=global-statement
    if not theme_name:
        theme_name = opts.gradio_theme
    if theme_name == "black-orange":
        gradio_theme = gr.themes.Default()
    elif theme_name.startswith("gradio/"):
        if theme_name == "gradio/default":
            gradio_theme = gr.themes.Default()
        if theme_name == "gradio/base":
            gradio_theme = gr.themes.Base()
        if theme_name == "gradio/glass":
            gradio_theme = gr.themes.Glass()
        if theme_name == "gradio/monochrome":
            gradio_theme = gr.themes.Monochrome()
        if theme_name == "gradio/soft":
            gradio_theme = gr.themes.Soft()
    else:
        try:
            gradio_theme = gr.themes.ThemeClass.from_hub(theme_name)
        except:
            print("Theme download error accessing HuggingFace")
            gradio_theme = gr.themes.Default()
    print(f'Loading theme: {theme_name}')


class TotalTQDM:
    def __init__(self):
        self._tqdm = None

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc="Total",
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


def restart_server():
    if demo is None:
        return
    try:
        import logging
        logging.disable(logging.CRITICAL)
        demo.server.should_exit = True
        demo.server.force_exit = True
        demo.close(verbose=False)
        demo.server.close()
    except:
        pass
    print('Server shutdown')


def listfiles(dirname):
    filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname), key=str.lower) if not x.startswith(".")]
    return [file for file in filenames if os.path.isfile(file)]


def html_path(filename):
    return os.path.join(paths.script_path, "html", filename)


def html(filename):
    path = html_path(filename)

    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()

    return ""
