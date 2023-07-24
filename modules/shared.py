import os
import sys
import time
import json
import datetime
import urllib.request
from urllib.parse import urlparse
from enum import Enum
import gradio as gr
import tqdm
import requests
import fasteners
from modules import errors, ui_components, shared_items, cmd_args
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir # pylint: disable=W0611
from modules.dml import directml_do_hijack, directml_override_opts
import modules.interrogate
import modules.memmon
import modules.styles
import modules.devices as devices # pylint: disable=R0402
import modules.paths_internal as paths
from installer import log as central_logger # pylint: disable=E0611


errors.install(gr)
demo: gr.Blocks = None
log = central_logger
progress_print_out = sys.stdout
parser = cmd_args.parser
url = 'https://github.com/vladmandic/automatic'
cmd_opts, _ = parser.parse_known_args()
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
xformers_available = False
clip_model = None
interrogator = modules.interrogate.InterrogateModels("interrogate")
sd_upscalers = []
face_restorers = []
tab_names = []
options_templates = {}
hypernetworks = {}
loaded_hypernetworks = []
gradio_theme = gr.themes.Base()
settings_components = None
pipelines = [
    'Stable Diffusion', 'Stable Diffusion XL', 'Kandinsky V1', 'Kandinsky V2', 'DeepFloyd IF', 'Shap-E',
    'Stable Diffusion Img2Img', 'Stable Diffusion XL Img2Img', 'Kandinsky V1 Img2Img', 'Kandinsky V2 Img2Img', 'DeepFloyd IF Img2Img', 'Shap-E Img2Img'
]
latent_upscale_default_mode = "Latent"
latent_upscale_modes = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}
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
ui_reorder_categories = [
    "inpaint",
    "sampler",
    "dimensions",
    "cfg",
    "seed",
    "batch",
    "checkboxes",
    "second_pass",
    "override_settings",
    "scripts",
]


def is_url(string):
    parsed_url = urlparse(string)
    return all([parsed_url.scheme, parsed_url.netloc])


class Backend(Enum):
    ORIGINAL = 1
    DIFFUSERS = 2


def reload_hypernetworks():
    from modules.hypernetworks import hypernetwork
    global hypernetworks # pylint: disable=W0603
    hypernetworks = hypernetwork.list_hypernetworks(opts.hypernetwork_dir)


class State:
    skipped = False
    interrupted = False
    paused = False
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
    oom = False

    def skip(self):
        log.debug('Requested skip')
        self.skipped = True

    def interrupt(self):
        log.debug('Requested interrupt')
        self.interrupted = True

    def pause(self):
        self.paused = not self.paused
        log.debug(f'Requested {"pause" if self.paused else "continue"}')

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
        self.paused = False
        self.textinfo = None
        self.time_start = time.time()
        devices.torch_gc()

    def end(self):
        self.job = ""
        self.job_count = 0
        self.paused = False
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
        try:
            if opts.show_progress_grid:
                self.assign_current_image(modules.sd_samplers.samples_to_image_grid(self.current_latent))
            else:
                self.assign_current_image(modules.sd_samplers.sample_to_image(self.current_latent))
        except Exception:
            pass
        self.current_image_sampling_step = self.sampling_step

    def assign_current_image(self, image):
        self.current_image = image
        self.id_live_preview += 1

state = State()
state.server_start = time.time()

backend = Backend.DIFFUSERS if (cmd_opts.backend is not None) and (cmd_opts.backend.lower() == 'diffusers') else Backend.ORIGINAL # initial since we don't have opts loaded yet


class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None, submit=None, comment_before='', comment_after=''):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh
        self.comment_before = comment_before # HTML text that will be added after label in UI
        self.comment_after = comment_after # HTML text that will be added before label in UI
        self.submit = submit

    def link(self, label, uri):
        self.comment_before += f"[<a href='{uri}' target='_blank'>{label}</a>]"
        return self

    def js(self, label, js_func):
        self.comment_before += f"[<a onclick='{js_func}(); return false'>{label}</a>]"
        return self

    def info(self, info):
        self.comment_after += f"<span class='info'>({info})</span>"
        return self

    def html(self, info):
        self.comment_after += f"<span class='info'>{info}</span>"
        return self

    def needs_restart(self):
        self.comment_after += " <span class='info'>(requires restart)</span>"
        return self


def options_section(section_identifier, options_dict):
    for v in options_dict.values():
        v.section = section_identifier
    return options_dict


def list_checkpoint_tiles():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.checkpoint_tiles()

default_checkpoint = list_checkpoint_tiles()[0] if len(list_checkpoint_tiles()) > 0 else "model.ckpt"


def refresh_checkpoints():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.list_models()


def list_samplers():
    import modules.sd_samplers # pylint: disable=W0621
    modules.sd_samplers.set_samplers()
    return modules.sd_samplers.all_samplers

def list_themes():
    fn = os.path.join('html', 'themes.json')
    if not os.path.exists(fn):
        refresh_themes()
    if os.path.exists(fn):
        with open(fn, mode='r', encoding='utf=8') as f:
            res = json.loads(f.read())
    else:
        res = []
    builtin = ["black-orange", "gradio/default", "gradio/base", "gradio/glass", "gradio/monochrome", "gradio/soft"]
    themes = sorted(set(builtin + [x['id'] for x in res if x['status'] == 'RUNNING' and 'test' not in x['id'].lower()]), key=str.casefold)
    return themes


def disable_extensions():
    if opts.lora_disable:
        if 'Lora' not in opts.disabled_extensions:
            opts.data['disabled_extensions'].append('Lora')
    else:
        opts.data['disabled_extensions'] = [x for x in opts.disabled_extensions if x != 'Lora']
    if backend == Backend.DIFFUSERS:
        for ext in ['sd-webui-controlnet', 'sd-dynamic-thresholding', 'multidiffusion-upscaler-for-automatic1111', 'a1111-sd-webui-lycoris']:
            if ext not in opts.disabled_extensions:
                log.warning(f'Diffusers disabling uncompatible extension: {ext}')
                opts.data['disabled_extensions'].append(ext)


def refresh_themes():
    try:
        req = requests.get('https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json', timeout=5)
        if req.status_code == 200:
            res = req.json()
            fn = os.path.join('html', 'themes.json')
            with open(fn, mode='w', encoding='utf=8') as f:
                f.write(json.dumps(res))
        else:
            log.error('Error refreshing UI themes')
    except Exception:
        log.error('Exception refreshing UI themes')


def readfile(filename, silent=False):
    data = {}
    try:
        if not os.path.exists(filename):
            return {}
        with fasteners.InterProcessLock(f"{filename}.lock"):
            with open(filename, "r", encoding="utf8") as file:
                data = json.load(file)
            if not silent:
                log.debug(f'Reading: {filename} len={len(data)}')
    except Exception as e:
        log.error(f'Reading failed: {filename} {e}')
    return data


def writefile(data, filename, mode='w'):

    def default(obj):
        log.error(f"Saving: {filename} not a valid object: {obj}")
        return str(obj)

    try:
        with fasteners.InterProcessLock(f"{filename}.lock"):
            # skipkeys=True, ensure_ascii=True, check_circular=True, allow_nan=True
            output = json.dumps(data, indent=2, default=default)
            log.debug(f'Saving: {filename} len={len(output)}')
            with open(filename, mode, encoding="utf8") as file:
                file.write(output)
    except Exception as e:
        log.error(f'Saving failed: {filename} {e}')


if devices.backend == "cpu":
    cross_attention_optimization_default = "Doggettx's"
elif devices.backend == "mps":
    cross_attention_optimization_default = "Doggettx's"
elif devices.backend == "ipex":
    cross_attention_optimization_default = "Sub-quadratic"
elif devices.backend == "directml":
    cross_attention_optimization_default = "Sub-quadratic"
elif devices.backend == "rocm":
    cross_attention_optimization_default = "Sub-quadratic"
else: # cuda
    cross_attention_optimization_default ="Scaled-Dot-Product"

options_templates.update(options_section(('sd', "Stable Diffusion"), {
    "sd_backend": OptionInfo("original", "Stable Diffusion backend", gr.Radio, lambda: {"choices": ["original", "diffusers"] }),
    "sd_checkpoint_autoload": OptionInfo(True, "Stable Diffusion checkpoint autoload on server start"),
    "sd_model_checkpoint": OptionInfo(default_checkpoint, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_model_refiner": OptionInfo('None', "Stable Diffusion refiner", gr.Dropdown, lambda: {"choices": ['None'] + list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_checkpoint_cache": OptionInfo(0, "Number of cached model checkpoints", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae_checkpoint_cache": OptionInfo(0, "Number of cached VAE checkpoints", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae": OptionInfo("Automatic", "Select VAE", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list),
    "sd_model_dict": OptionInfo('None', "Stable Diffusion checkpoint dict", gr.Dropdown, lambda: {"choices": ['None'] + list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "stream_load": OptionInfo(False, "Load models using stream loading method"),
    "model_reuse_dict": OptionInfo(False, "When loading models attempt to reuse previous model dictionary"),
    "prompt_attention": OptionInfo("Full parser", "Prompt attention parser", gr.Radio, lambda: {"choices": ["Full parser", "Compel parser", "A1111 parser", "Fixed attention"] }),
    "prompt_mean_norm": OptionInfo(True, "Prompt attention mean normalization"),
    "comma_padding_backtrack": OptionInfo(20, "Prompt padding for long prompts", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1 }),
    "sd_disable_ckpt": OptionInfo(False, "Disallow usage of checkpoints in ckpt format"),
}))

options_templates.update(options_section(('optimizations', "Optimizations"), {
    "cross_attention_optimization": OptionInfo(cross_attention_optimization_default, "Cross-attention optimization method", gr.Radio, lambda: {"choices": shared_items.list_crossattention() }),
    "cross_attention_options": OptionInfo([], "Cross-attention advanced options", gr.CheckboxGroup, lambda: {"choices": ['xFormers enable flash Attention', 'SDP disable memory attention']}),
    "sub_quad_q_chunk_size": OptionInfo(512, "Sub-quadratic cross-attention query chunk size", gr.Slider, {"minimum": 16, "maximum": 8192, "step": 8}),
    "sub_quad_kv_chunk_size": OptionInfo(512, "Sub-quadratic cross-attention kv chunk size", gr.Slider, {"minimum": 0, "maximum": 8192, "step": 8}),
    "sub_quad_chunk_threshold": OptionInfo(80, "Sub-quadratic cross-attention chunking threshold", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),
    "token_merging_ratio": OptionInfo(0.0, "Token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}),
    "token_merging_ratio_img2img": OptionInfo(0.0, "Token merging ratio for img2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}),
    "token_merging_ratio_hr": OptionInfo(0.0, "Token merging ratio for hires pass", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}),
    "sd_vae_sliced_encode": OptionInfo(False, "Enable splitting of hires batch processing"),
}))

options_templates.update(options_section(('cuda', "Compute Settings"), {
    "memmon_poll_rate": OptionInfo(2, "VRAM usage polls per second during generation", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}),
    "precision": OptionInfo("Autocast", "Precision type", gr.Radio, lambda: {"choices": ["Autocast", "Full"]}),
    "cuda_dtype": OptionInfo("FP32" if sys.platform == "darwin" else "FP16", "Device precision type", gr.Radio, lambda: {"choices": ["FP32", "FP16", "BF16"]}),
    "no_half": OptionInfo(False, "Use full precision for model (--no-half)", None, None, None),
    "no_half_vae": OptionInfo(False, "Use full precision for VAE (--no-half-vae)"),
    "upcast_sampling": OptionInfo(True if sys.platform == "darwin" else False, "Enable upcast sampling"),
    "upcast_attn": OptionInfo(False, "Enable upcast cross attention layer"),
    "cuda_cast_unet": OptionInfo(False, "Use fixed UNet precision"),
    "disable_nan_check": OptionInfo(True, "Disable NaN check in produced images/latent spaces"),
    "rollback_vae": OptionInfo(False, "Attempt VAE roll back when produced NaN values (experimental)"),
    "opt_channelslast": OptionInfo(False, "Use channels last as torch memory format "),
    "cudnn_benchmark": OptionInfo(False, "Enable full-depth cuDNN benchmark feature"),
    "cuda_allow_tf32": OptionInfo(True, "Allow TF32 math ops"),
    "cuda_allow_tf16_reduced": OptionInfo(True, "Allow TF16 reduced precision math ops"),
    "cuda_compile": OptionInfo(False, "Enable model compile (experimental)"),
    "cuda_compile_mode": OptionInfo("none", "Model compile mode (experimental)", gr.Radio, lambda: {"choices": ['none', 'inductor', 'reduce-overhead', 'cudagraphs', 'aot_ts_nvfuser', 'hidet', 'ipex']}),
    "cuda_compile_fullgraph": OptionInfo(False, "Model compile fullgraph"),
    "cuda_compile_verbose": OptionInfo(False, "Model compile verbose mode"),
    "cuda_compile_errors": OptionInfo(True, "Model compile suppress errors"),
    "disable_gc": OptionInfo(True, "Disable Torch memory garbage collection"),
}))

options_templates.update(options_section(('diffusers', "Diffusers Settings"), {
    "diffusers_allow_safetensors": OptionInfo(True, 'Diffusers allow loading from safetensors files'),
    "diffusers_pipeline": OptionInfo(pipelines[0], 'Diffusers pipeline', gr.Dropdown, lambda: {"choices": pipelines}),
    "diffusers_refiner_latents": OptionInfo(True, "Use latents when using refiner"),
    "diffusers_move_base": OptionInfo(False, "Move base model to CPU when using refiner"),
    "diffusers_move_refiner": OptionInfo(True, "Move refiner model to CPU when not in use"),
    "diffusers_extract_ema": OptionInfo(True, "Use model EMA weights when possible"),
    "diffusers_generator_device": OptionInfo("default", "Generator device", gr.Radio, lambda: {"choices": ["default", "cpu"]}),
    "diffusers_seq_cpu_offload": OptionInfo(False, "Enable sequential CPU offload"),
    "diffusers_model_cpu_offload": OptionInfo(False, "Enable model CPU offload"),
    "diffusers_vae_upcast": OptionInfo("default", "VAE upcasting", gr.Radio, lambda: {"choices": ['default', 'true', 'false']}),
    "diffusers_vae_slicing": OptionInfo(True, "Enable VAE slicing"),
    "diffusers_vae_tiling": OptionInfo(False, "Enable VAE tiling"),
    "diffusers_attention_slicing": OptionInfo(False, "Enable attention slicing"),
    # "diffusers_force_zeros": OptionInfo(False, "Force zeros for prompts when empty"),
    # "diffusers_aesthetics_score": OptionInfo(6.0, "Require aesthetic score", gr.Slider, {"minimum": 0, "maximum": 10, "step": 0.1}),
}))

options_templates.update(options_section(('system-paths', "System Paths"), {
    "temp_dir": OptionInfo("", "Directory for temporary images; leave empty for default"),
    "clean_temp_dir_at_start": OptionInfo(True, "Cleanup non-default temporary directory when starting webui"),
    "ckpt_dir": OptionInfo(os.path.join(paths.models_path, 'Stable-diffusion'), "Path to directory with stable diffusion checkpoints"),
    "diffusers_dir": OptionInfo(os.path.join(paths.models_path, 'Diffusers'), "Path to directory with stable diffusion diffusers"),
    "vae_dir": OptionInfo(os.path.join(paths.models_path, 'VAE'), "Path to directory with VAE files"),
    "lora_dir": OptionInfo(os.path.join(paths.models_path, 'Lora'), "Path to directory with Lora network(s)"),
    "lyco_dir": OptionInfo(os.path.join(paths.models_path, 'LyCORIS'), "Path to directory with LyCORIS network(s)"),
    "styles_dir": OptionInfo(os.path.join(paths.data_path, 'styles.csv'), "Path to user-defined styles file"),
    "embeddings_dir": OptionInfo(os.path.join(paths.models_path, 'embeddings'), "Embeddings directory for textual inversion"),
    "hypernetwork_dir": OptionInfo(os.path.join(paths.models_path, 'hypernetworks'), "Hypernetwork directory"),
    "codeformer_models_path": OptionInfo(os.path.join(paths.models_path, 'Codeformer'), "Path to directory with codeformer model file(s)"),
    "gfpgan_models_path": OptionInfo(os.path.join(paths.models_path, 'GFPGAN'), "Path to directory with GFPGAN model file(s)"),
    "esrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'ESRGAN'), "Path to directory with ESRGAN model file(s)"),
    "bsrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'BSRGAN'), "Path to directory with BSRGAN model file(s)"),
    "realesrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'RealESRGAN'), "Path to directory with RealESRGAN model file(s)"),
    "scunet_models_path": OptionInfo(os.path.join(paths.models_path, 'ScuNET'), "Path to directory with ScuNET model file(s)"),
    "swinir_models_path": OptionInfo(os.path.join(paths.models_path, 'SwinIR'), "Path to directory with SwinIR model file(s)"),
    "ldsr_models_path": OptionInfo(os.path.join(paths.models_path, 'LDSR'), "Path to directory with LDSR model file(s)"),
    "clip_models_path": OptionInfo(os.path.join(paths.models_path, 'CLIP'), "Path to directory with CLIP model file(s)"),
}))

options_templates.update(options_section(('saving-images', "Image Options"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('jpg', 'File format for generated images', gr.Dropdown, lambda: {"choices": ["jpg", "png", "webp", "tiff", "jp2"]}),
    "image_metadata": OptionInfo(True, "Include metadata in saved images"),
    "image_watermark_enabled": OptionInfo(False, "Include watermark in saved images"),
    "image_watermark": OptionInfo('', "Image watermark string"),
    "samples_filename_pattern": OptionInfo("[seq]-[prompt_words]", "Images filename pattern", component_args=hide_dirs),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 99, "step": 1, **hide_dirs}),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),
    "samples_save_zip": OptionInfo(True, "Create zip archive when downloading multiple images"),
    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('jpg', 'File format for grids', gr.Dropdown, lambda: {"choices": ["jpg", "png", "webp", "tiff", "jp2"]}),
    "grid_extended_filename": OptionInfo(True, "Add extended info (seed, prompt) to filename when saving grid"),
    "grid_only_if_multiple": OptionInfo(True, "Do not save grids consisting of one picture"),
    "grid_prevent_empty_spots": OptionInfo(True, "Prevent empty spots in grid (when set to autodetect)"),
    "n_rows": OptionInfo(-1, "Grid row count", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    "save_txt": OptionInfo(False, "Create text file next to every image with generation parameters"),
    "save_log_fn": OptionInfo("", "Create JSON log file for each saved image", component_args=hide_dirs),
    "save_images_before_highres_fix": OptionInfo(False, "Save copy of image before applying highres fix"),
    "save_images_before_refiner": OptionInfo(False, "Save copy of image before running refiner"),
    "save_images_before_face_restoration": OptionInfo(False, "Save copy of image before doing face restoration"),
    "save_images_before_color_correction": OptionInfo(False, "Save copy of image before applying color correction"),
    "save_mask": OptionInfo(False, "Save copy of the inpainting greyscale mask"),
    "save_mask_composite": OptionInfo(False, "Save copy of inpainting masked composite"),
    "save_init_img": OptionInfo(False, "Save copy of processing init images"),
    "jpeg_quality": OptionInfo(85, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
    "img_max_size_mp": OptionInfo(250, "Maximum allowed image size in megapixels", gr.Number),
    "use_original_name_batch": OptionInfo(True, "Use original name for output filename during batch process"),
    "use_upscaler_name_as_suffix": OptionInfo(True, "Use upscaler name as filename suffix in the extras tab"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "save_to_dirs": OptionInfo(False, "Save images to a subdirectory"),
    "grid_save_to_dirs": OptionInfo(False, "Save grids to a subdirectory"),
    "use_save_to_dirs_for_ui": OptionInfo(False, "Save images to a subdirectory when using Save button"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs),
}))

options_templates.update(options_section(('image-processing', "Image Processing"), {
    "img2img_color_correction": OptionInfo(False, "Apply color correction to match original colors"),
    "img2img_fix_steps": OptionInfo(False, "For image processing do exact number of steps as specified"),
    "img2img_background_color": OptionInfo("#ffffff", "Image transparent color fill", ui_components.FormColorPicker, {}),
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for image processing", gr.Slider, {"minimum": 0.1, "maximum": 1.5, "step": 0.01}),
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 8, "step": 1, "visible": False}),
}))


options_templates.update(options_section(('saving-paths', "Image Paths"), {
    "outdir_samples": OptionInfo("", "Output directory for images", component_args=hide_dirs),
    "outdir_txt2img_samples": OptionInfo("outputs/text", 'Output directory for txt2img images', component_args=hide_dirs),
    "outdir_img2img_samples": OptionInfo("outputs/image", 'Output directory for img2img images', component_args=hide_dirs),
    "outdir_extras_samples": OptionInfo("outputs/extras", 'Output directory for images from extras tab', component_args=hide_dirs),
    "outdir_grids": OptionInfo("", "Output directory for grids", component_args=hide_dirs),
    "outdir_txt2img_grids": OptionInfo("outputs/grids", 'Output directory for txt2img grids', component_args=hide_dirs),
    "outdir_img2img_grids": OptionInfo("outputs/grids", 'Output directory for img2img grids', component_args=hide_dirs),
    "outdir_save": OptionInfo("outputs/save", "Directory for saving images using the Save button", component_args=hide_dirs),
    "outdir_init_images": OptionInfo("outputs/init-images", "Directory for saving init images when using img2img", component_args=hide_dirs),
}))

options_templates.update(options_section(('ui', "User Interface"), {
    "gradio_theme": OptionInfo("black-orange", "UI theme", gr.Dropdown, lambda: {"choices": list_themes()}, refresh=refresh_themes),
    "theme_style": OptionInfo("Auto", "Theme mode", gr.Radio, {"choices": ["Auto", "Dark", "Light"]}),
    "tooltips": OptionInfo("UI Tooltips", "UI tooltips", gr.Radio, {"choices": ["None", "Browser default", "UI tooltips"]}),
    "return_grid": OptionInfo(True, "Show grid in results for web"),
    "return_mask": OptionInfo(False, "For inpainting, include the greyscale mask in results for web"),
    "return_mask_composite": OptionInfo(False, "For inpainting, include masked composite in results for web"),
    "disable_weights_auto_swap": OptionInfo(True, "Do not change selected model when reading generation parameters"),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
    "font": OptionInfo("", "Font for image grids that have text"),
    "keyedit_precision_attention": OptionInfo(0.1, "Ctrl+up/down precision when editing (attention:1.1)", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_precision_extra": OptionInfo(0.05, "Ctrl+up/down precision when editing <extra networks:0.9>", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001}),
    "keyedit_delimiters": OptionInfo(".,\/!?%^*;:{}=`~()", "Ctrl+up/down word delimiters"), # pylint: disable=anomalous-backslash-in-string
    "quicksettings_list": OptionInfo(["sd_model_checkpoint"], "Quicksettings list", ui_components.DropdownMulti, lambda: {"choices": list(opts.data_labels.keys())}),
    "hidden_tabs": OptionInfo([], "Hidden UI tabs", ui_components.DropdownMulti, lambda: {"choices": list(tab_names)}),
    "ui_tab_reorder": OptionInfo("From Text, From Image, Process Image", "UI tabs order"),
    "ui_scripts_reorder": OptionInfo("Enable Dynamic Thresholding, ControlNet", "UI scripts order"),
    "ui_reorder": OptionInfo(", ".join(ui_reorder_categories), "txt2img/img2img UI item order"),
}))

options_templates.update(options_section(('live-preview', "Live Previews"), {
    "show_progressbar": OptionInfo(True, "Show progressbar"),
    "live_previews_enable": OptionInfo(True, "Show live previews of the created image"),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid"),
    "notification_audio_enable": OptionInfo(False, "Play a sound when images are finished generating"),
    "notification_audio_path": OptionInfo("html/notification.mp3","Path to notification sound", component_args=hide_dirs),
    "show_progress_every_n_steps": OptionInfo(1, "Live preview display period", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "show_progress_type": OptionInfo("Approximate NN", "Live preview method", gr.Radio, {"choices": ["Full VAE", "Approximate NN", "Approximate simple", "TAESD"]}),
    "live_preview_content": OptionInfo("Combined", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"]}),
    "live_preview_refresh_period": OptionInfo(500, "Progressbar/preview update period, in milliseconds", gr.Slider, {"minimum": 0, "maximum": 5000, "step": 25}),
    "logmonitor_show": OptionInfo(True, "Show log view"),
    "logmonitor_refresh_period": OptionInfo(5000, "Log view update period, in milliseconds", gr.Slider, {"minimum": 0, "maximum": 30000, "step": 25}),
}))

options_templates.update(options_section(('sampler-params', "Sampler Settings"), {
    "show_samplers": OptionInfo(["Default", "Euler a", "UniPC", "DEIS", "DDIM", "DPM 1S", "DPM 2M", "DPM++ 2M SDE", "DPM++ 2M SDE Karras", "DPM2 Karras", "DPM++ 2M Karras"], "Show samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in list_samplers() if x.name != "PLMS"]}),
    # "fallback_sampler": OptionInfo("Euler a", "Secondary sampler", gr.Dropdown, lambda: {"choices": ["None"] + [x.name for x in list_samplers()]}),
    # "force_latent_sampler": OptionInfo("None", "Force latent upscaler sampler", gr.Dropdown, lambda: {"choices": ["None"] + [x.name for x in list_samplers()]}),
    'uni_pc_variant': OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}),
    'eta_noise_seed_delta': OptionInfo(0, "Noise seed delta (eta)", gr.Number, {"precision": 0}),
    "eta_ddim": OptionInfo(0.0, "Noise multiplier for DDIM (eta)", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "schedulers_solver_order": OptionInfo(2, "Samplers solver order where applicable", gr.Slider, {"minimum": 1, "maximum": 5, "step": 1}),

    "schedulers_sep_diffusers": OptionInfo("<h2>Diffusers specific config</h2>", "", gr.HTML),
    "schedulers_prediction_type": OptionInfo("default", "Samplers override model prediction type", gr.Radio, lambda: {"choices": ['default', 'epsilon', 'sample', 'v-prediction']}),
    "schedulers_use_karras": OptionInfo(True, "Samplers should use Karras sigmas where applicable"),
    "schedulers_use_loworder": OptionInfo(True, "Samplers should use use lower-order solvers in the final steps where applicable"),
    "schedulers_use_thresholding": OptionInfo(False, "Samplers should use dynamic thresholding where applicable"),
    "schedulers_dpm_solver": OptionInfo("sde-dpmsolver++", "Samplers DPM solver algorithm", gr.Radio, lambda: {"choices": ['dpmsolver', 'dpmsolver++', 'sde-dpmsolver++']}),
    "schedulers_beta_schedule": OptionInfo("default", "Samplers override beta schedule", gr.Radio, lambda: {"choices": ['default', 'linear', 'scaled_linear', 'squaredcos_cap_v2']}),
    'schedulers_beta_start': OptionInfo(0, "Samplers override beta start", gr.Number, {}),
    'schedulers_beta_end': OptionInfo(0, "Samplers override beta end", gr.Number, {}),

    "schedulers_sep_kdiffusers": OptionInfo("<h2>K-Diffusion specific config</h2>", "", gr.HTML),
    "always_batch_cond_uncond": OptionInfo(False, "Disable conditional batching enabled on low memory systems"),
    "enable_quantization": OptionInfo(True, "Enable samplers quantization for sharper and cleaner results"),
    "eta_ancestral": OptionInfo(1.0, "Noise multiplier for ancestral samplers (eta)", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_min_uncond': OptionInfo(0, "sigma negative guidance minimum ", gr.Slider, {"minimum": 0.0, "maximum": 4.0, "step": 0.01}),
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    'always_discard_next_to_last_sigma': OptionInfo(False, "Always discard next-to-last sigma"),

    "schedulers_sep_compvis": OptionInfo("<h2>CompVis specific config</h2>", "", gr.HTML),
    "ddim_discretize": OptionInfo('uniform', "DDIM discretize img2img", gr.Radio, {"choices": ['uniform', 'quad']}),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Enable addtional postprocessing operations", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'upscaling_max_images_in_cache': OptionInfo(5, "Maximum number of images in upscaling cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
}))

options_templates.update(options_section(('training', "Training"), {
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training if possible"),
    "pin_memory": OptionInfo(True, "Pin training dataset to memory"),
    "save_optimizer_state": OptionInfo(False, "Save resumable optimizer state when training"),
    "save_training_settings_to_txt": OptionInfo(True, "Save training settings to a text file on training start"),
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex"),
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string"),
    "embeddings_templates_dir": OptionInfo(os.path.join(paths.script_path, 'train', 'templates'), "Embeddings train templates directory"),
    "training_image_repeats_per_epoch": OptionInfo(1, "Number of repeats for a single input image per epoch", gr.Number, {"precision": 0}),
    "training_write_csv_every": OptionInfo(0, "Save CSV file containing the loss to log directory"),
    "training_enable_tensorboard": OptionInfo(False, "Enable tensorboard logging"),
    "training_tensorboard_save_images": OptionInfo(False, "Save generated images within tensorboard"),
    "training_tensorboard_flush_every": OptionInfo(120, "Tensorboard flush period"),
}))

options_templates.update(options_section(('interrogate', "Interrogate"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Interrogate: keep models in VRAM"),
    "interrogate_return_ranks": OptionInfo(True, "Interrogate: include ranks of model tags matches in results"),
    "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(32, "Interrogate: minimum description length", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(192, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(2048, "CLIP: maximum number of lines in text file"),
    "interrogate_clip_skip_categories": OptionInfo(["artists", "movements", "flavors"], "CLIP: skip inquire categories", gr.CheckboxGroup, lambda: {"choices": modules.interrogate.category_types()}, refresh=modules.interrogate.category_types),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.65, "Interrogate: deepbooru score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(False, "Interrogate: deepbooru sort alphabetically"),
    "deepbooru_use_spaces": OptionInfo(False, "Use spaces for tags in deepbooru"),
    "deepbooru_escape": OptionInfo(True, "Escape brackets in deepbooru"),
    "deepbooru_filter_tags": OptionInfo("", "Filter out tags from deepbooru output"),
}))

options_templates.update(options_section(('upscaling', "Upscaling"), {
    "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.2, "CodeFormer weight parameter", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
    "upscaler_for_img2img": OptionInfo("None", "Default upscaler for image resize operations", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN 4x+", "R-ESRGAN 4x+ Anime6B"], "Real-ESRGAN available models", gr.CheckboxGroup, lambda: {"choices": shared_items.realesrgan_models_names()}),
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN upscalers", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap in pixels for ESRGAN upscalers", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "SCUNET_tile": OptionInfo(256, "Tile size for SCUNET upscalers", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "SCUNET_tile_overlap": OptionInfo(8, "Tile overlap for SCUNET upscalers", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}),
    "use_old_hires_fix_width_height": OptionInfo(False, "Hires fix uses width & height to set final resolution"),
    "dont_fix_second_order_samplers_schedule": OptionInfo(False, "Do not fix prompt schedule for second order samplers"),
}))

options_templates.update(options_section(('extra_networks', "Extra Networks"), {
    "ui_extra_networks_tab_reorder": OptionInfo("Checkpoints, Lora, LyCORIS, Textual Inversion, Hypernetworks", "Extra networks tab order"),
    "extra_networks_card_cover": OptionInfo("sidebar", "UI position", gr.Radio, lambda: {"choices": ["cover", "inline", "sidebar"]}),
    "extra_networks_height": OptionInfo(53, "UI height (%)", gr.Slider, {"minimum": 10, "maximum": 100, "step": 1}),
    "extra_networks_sidebar_width": OptionInfo(35, "UI sidebar width (%)", gr.Slider, {"minimum": 10, "maximum": 80, "step": 1}),
    "extra_networks_card_lazy": OptionInfo(True, "UI card preview lazy loading"),
    "extra_networks_card_size": OptionInfo(200, "UI card size (px)", gr.Slider, {"minimum": 20, "maximum": 2000, "step": 1}),
    "extra_networks_card_square": OptionInfo(True, "UI disable variable aspect ratio"),
    "extra_networks_card_fit": OptionInfo("cover", "UI image contain method", gr.Radio, lambda: {"choices": ["contain", "cover", "fill"]}),
    "extra_network_skip_indexing": OptionInfo(False, "Do not automatically build extra network pages", gr.Checkbox),
    "lyco_patch_lora": OptionInfo(False, "Use LyCoris handler for all Lora types", gr.Checkbox),
    "lora_disable": OptionInfo(False, "Disable built-in Lora handler", gr.Checkbox, { "visible": True }, onchange=disable_extensions),
    "lora_functional": OptionInfo(False, "Use Kohya method for handling multiple Loras", gr.Checkbox),
    "extra_networks_add_text_separator": OptionInfo(" ", "Extra text to add before <...> when adding extra network to prompt", gr.Text, { "visible": False }),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, lambda: {"choices": ["None"] + list(hypernetworks.keys())}, refresh=reload_hypernetworks),
}))

options_templates.update(options_section((None, "Hidden options"), {
    "disabled_extensions": OptionInfo([], "Disable these extensions"),
    "disable_all_extensions": OptionInfo("none", "Disable all extensions (preserves the list of disabled extensions)", gr.Radio, {"choices": ["none", "user", "all"]}),
    "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint"),
}))

options_templates.update()


class Options:
    data = None
    data_labels = options_templates
    typemap = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value): # pylint: disable=inconsistent-return-statements
        if self.data is not None:
            if key in self.data or key in self.data_labels:
                if cmd_opts.freeze:
                    log.warning(f'Settings are frozen: {key}')
                    return
                if cmd_opts.hide_ui_dir_config and key in restricted_opts:
                    log.warning(f'Settings key is restricted: {key}')
                    return
                self.data[key] = value
                return
        return super(Options, self).__setattr__(key, value) # pylint: disable=super-with-arguments

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]
        if item in self.data_labels:
            return self.data_labels[item].default
        return super(Options, self).__getattribute__(item) # pylint: disable=super-with-arguments

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
                log.error(f'Error in onchange callback: {key} {value} {e}')
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
        if cmd_opts.freeze:
            log.warning(f'Settings saving is disabled: {filename}')
            return
        try:
            output = json.dumps(self.data, indent=2)
            log.debug(f'Saving settings: {filename} len={len(output)}')
            with open(filename, "w", encoding="utf8") as file:
                file.write(output)
        except Exception as e:
            log.error(f'Saving settings failed: {filename} {e}')

    def same_type(self, x, y):
        if x is None or y is None:
            return True
        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))
        return type_x == type_y

    def load(self, filename):
        if not os.path.isfile(filename):
            log.debug(f'Created default config: {filename}')
            self.save(filename)
            return
        self.data = readfile(filename)
        if self.data.get('quicksettings') is not None and self.data.get('quicksettings_list') is None:
            self.data['quicksettings_list'] = [i.strip() for i in self.data.get('quicksettings').split(',')]
        bad_settings = 0
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                log.error(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})")
                bad_settings += 1
        if bad_settings > 0:
            log.error(f"Error: Bad settings found in {filename}")

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func
        if call:
            func()

    def dumpjson(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        metadata = {
            k: {
                "is_stored": k in self.data,
                "tab_name": v.section[0]
            } for k, v in self.data_labels.items()
        }
        return json.dumps({"values": d, "metadata": metadata})

    def add_option(self, key, info):
        self.data_labels[key] = info

    def reorder(self):
        """reorder settings so that all items related to section always go together"""
        section_ids = {}
        settings_items = self.data_labels.items()
        for _k, item in settings_items:
            if item.section not in section_ids:
                section_ids[item.section] = len(section_ids)
        self.data_labels = dict(sorted(settings_items, key=lambda x: section_ids[x[1].section]))

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
        elif expected_type == type(value):
            pass
        else:
            value = expected_type(value)
        return value


opts = Options()
config_filename = cmd_opts.config
opts.load(config_filename)
cmd_opts = cmd_args.compatibility_args(opts, cmd_opts)
if cmd_opts.backend is None:
    backend = Backend.DIFFUSERS if opts.data.get('sd_backend', 'original') == 'diffusers' else Backend.ORIGINAL
else:
    backend = Backend.DIFFUSERS if cmd_opts.backend.lower() == 'diffusers' else Backend.ORIGINAL
opts.data['sd_backend'] = 'diffusers' if backend == Backend.DIFFUSERS else 'original'
opts.data['uni_pc_lower_order_final'] = opts.schedulers_use_loworder
opts.data['uni_pc_order'] = opts.schedulers_solver_order
log.info(f'Pipeline: {backend}')


prompt_styles = modules.styles.StyleDatabase(opts.styles_dir)
cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or (cmd_opts.server_name or False)) and not cmd_opts.insecure
devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])
device = devices.device
batch_cond_uncond = opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram
mem_mon = modules.memmon.MemUsageMonitor("MemMon", device, opts)
mem_mon.start()
if devices.backend == "directml":
    directml_do_hijack()
    directml_override_opts()


def reload_gradio_theme(theme_name=None):
    global gradio_theme # pylint: disable=global-statement
    if not theme_name:
        theme_name = opts.gradio_theme
    default_font_params = {}
    res = 0
    try:
        req = urllib.request.Request("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono", method="HEAD")
        res = urllib.request.urlopen(req, timeout=3.0).status # pylint: disable=consider-using-with
    except Exception:
        res = 0
    if res != 200:
        log.info('No internet access detected, using default fonts')
        default_font_params = {
            'font':['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
            'font_mono':['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace']
        }
    if theme_name == "black-orange":
        gradio_theme = gr.themes.Default(**default_font_params)
    elif theme_name.startswith("gradio/"):
        if theme_name == "gradio/default":
            gradio_theme = gr.themes.Default(**default_font_params)
        if theme_name == "gradio/base":
            gradio_theme = gr.themes.Base(**default_font_params)
        if theme_name == "gradio/glass":
            gradio_theme = gr.themes.Glass(**default_font_params)
        if theme_name == "gradio/monochrome":
            gradio_theme = gr.themes.Monochrome(**default_font_params)
        if theme_name == "gradio/soft":
            gradio_theme = gr.themes.Soft(**default_font_params)
    else:
        try:
            gradio_theme = gr.themes.ThemeClass.from_hub(theme_name)
        except Exception:
            log.error("Theme download error accessing HuggingFace")
            gradio_theme = gr.themes.Default(**default_font_params)
    log.info(f'Loading UI theme: name={theme_name} style={opts.theme_style}')


class TotalTQDM:
    def __init__(self):
        self._tqdm = None

    def reset(self):
        self._tqdm = tqdm.tqdm(
            desc="Total",
            total=state.job_count * state.sampling_steps,
            position=1,
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


def restart_server(restart=True):
    if demo is None:
        return
    log.info('Server shutdown requested')
    try:
        demo.server.wants_restart = restart
        demo.server.should_exit = True
        demo.server.force_exit = True
        demo.close(verbose=False)
        demo.server.close()
        demo.fns = []
        # os._exit(0)
    except Exception as e:
        log.error(f'Server shutdown error: {e}')
    if restart:
        log.info('Server will restart')


def restore_defaults(restart=True):
    if os.path.exists(cmd_opts.config):
        log.info('Restoring server defaults')
        os.remove(cmd_opts.config)
    restart_server(restart)


def listfiles(dirname):
    filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname), key=str.lower) if not x.startswith(".")]
    return [file for file in filenames if os.path.isfile(file)]


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return
    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)
    for root, _dirs, files in os.walk(path, followlinks=True):
        for filename in files:
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue
            yield os.path.join(root, filename)


def html_path(filename):
    return os.path.join(paths.script_path, "html", filename)


def html(filename):
    path = html_path(filename)
    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()
    return ""


def get_version():
    version = None
    if version is None:
        try:
            import subprocess
            res = subprocess.run('git log --pretty=format:"%h %ad" -1 --date=short', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            ver = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else '  '
            githash, updated = ver.split(' ')
            res = subprocess.run('git remote get-url origin', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            origin = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
            res = subprocess.run('git rev-parse --abbrev-ref HEAD', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            branch = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
            version = {
                'app': 'sd.next',
                'updated': updated,
                'hash': githash,
                'url': origin.replace('\n', '') + '/tree/' + branch.replace('\n', '')
            }
        except Exception:
            version = { 'app': 'sd.next' }
    return version


class Shared(sys.modules[__name__].__class__): # this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than at program startup.
    @property
    def sd_model(self):
        import modules.sd_models # pylint: disable=W0621
        return modules.sd_models.model_data.get_sd_model()

    @sd_model.setter
    def sd_model(self, value):
        import modules.sd_models # pylint: disable=W0621
        modules.sd_models.model_data.set_sd_model(value)

    @property
    def sd_refiner(self):
        import modules.sd_models # pylint: disable=W0621
        return modules.sd_models.model_data.get_sd_refiner()

    @sd_refiner.setter
    def sd_refiner(self, value):
        import modules.sd_models # pylint: disable=W0621
        modules.sd_models.model_data.set_sd_refiner(value)

    @property
    def backend(self):
        return Backend.ORIGINAL if opts.data['sd_backend'] == 'original' else Backend.DIFFUSERS


sd_model = None
sd_refiner = None
sys.modules[__name__].__class__ = Shared
