import io
import os
import sys
import time
import json
import contextlib
from types import SimpleNamespace
from urllib.parse import urlparse
from enum import Enum
import requests
import gradio as gr
import fasteners
from rich.console import Console
from modules import errors, shared_items, shared_state, cmd_args, ui_components, theme
from modules.paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir # pylint: disable=W0611
from modules.dml import memory_providers, default_memory_provider, directml_do_hijack
import modules.interrogate
import modules.memmon
import modules.styles
import modules.devices as devices # pylint: disable=R0402
import modules.paths_internal as paths
from installer import print_dict
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
extra_networks = []
options_templates = {}
hypernetworks = {}
loaded_hypernetworks = []
settings_components = None
latent_upscale_default_mode = "None"
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
compatibility_opts = ['clip_skip', 'uni_pc_lower_order_final', 'uni_pc_order']
console = Console(log_time=True, log_time_format='%H:%M:%S-%f')


class Backend(Enum):
    ORIGINAL = 1
    DIFFUSERS = 2


state = shared_state.State()
if not hasattr(cmd_opts, "use_openvino"):
    cmd_opts.use_openvino = False
if cmd_opts.use_openvino:
    backend = Backend.DIFFUSERS
    cmd_opts.backend = 'diffusers'
else:
    backend = Backend.DIFFUSERS if (cmd_opts.backend is not None) and (cmd_opts.backend.lower() == 'diffusers') else Backend.ORIGINAL # initial since we don't have opts loaded yet


class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None, folder=None, submit=None, comment_before='', comment_after=''):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh
        self.folder = folder
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


def is_url(string):
    parsed_url = urlparse(string)
    return all([parsed_url.scheme, parsed_url.netloc])


def reload_hypernetworks():
    from modules.hypernetworks import hypernetwork
    global hypernetworks # pylint: disable=W0603
    hypernetworks = hypernetwork.list_hypernetworks(opts.hypernetwork_dir)


def refresh_checkpoints():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.list_models()


def refresh_vaes():
    import modules.sd_vae # pylint: disable=W0621
    modules.sd_vae.refresh_vae_list()


def list_samplers():
    import modules.sd_samplers # pylint: disable=W0621
    modules.sd_samplers.set_samplers()
    return modules.sd_samplers.all_samplers


def temp_disable_extensions():
    disabled = []
    if cmd_opts.safe:
        for ext in ['sd-webui-controlnet', 'multidiffusion-upscaler-for-automatic1111', 'a1111-sd-webui-lycoris', 'sd-webui-agent-scheduler', 'clip-interrogator-ext', 'stable-diffusion-webui-rembg', 'sd-extension-chainner', 'stable-diffusion-webui-images-browser']:
            if ext not in opts.disabled_extensions:
                disabled.append(ext)
        log.info(f'Safe mode disabling extensions: {disabled}')
    if backend == Backend.DIFFUSERS:
        for ext in ['sd-webui-controlnet', 'multidiffusion-upscaler-for-automatic1111', 'a1111-sd-webui-lycoris']:
            if ext not in opts.disabled_extensions:
                disabled.append(ext)
        log.info(f'Diffusers disabling uncompatible extensions: {disabled}')
    cmd_opts.controlnet_loglevel = 'WARNING'
    return disabled


def readfile(filename, silent=False):
    data = {}
    try:
        if not os.path.exists(filename):
            return {}
        with fasteners.InterProcessLock(f"{filename}.lock"):
            with open(filename, "r", encoding="utf8") as file:
                data = json.load(file)
                if type(data) is str:
                    data = json.loads(data)
            if not silent:
                log.debug(f'Reading: {filename} len={len(data)}')
    except Exception as e:
        if not silent:
            log.error(f'Reading failed: {filename} {e}')
        return {}
    return data


def writefile(data, filename, mode='w', silent=False):
    def default(obj):
        log.error(f"Saving: {filename} not a valid object: {obj}")
        return str(obj)

    try:
        with fasteners.InterProcessLock(f"{filename}.lock"):
            # skipkeys=True, ensure_ascii=True, check_circular=True, allow_nan=True
            if type(data) == dict:
                output = json.dumps(data, indent=2, default=default)
            elif type(data) == list:
                output = json.dumps(data, indent=2, default=default)
            elif isinstance(data, object):
                simple = {}
                for k in data.__dict__:
                    if data.__dict__[k] is not None:
                        simple[k] = data.__dict__[k]
                output = json.dumps(simple, indent=2, default=default)
            else:
                raise ValueError('not a valid object')
            if not silent:
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
    cross_attention_optimization_default = "Scaled-Dot-Product"
elif devices.backend == "directml":
    cross_attention_optimization_default = "Sub-quadratic"
elif devices.backend == "rocm":
    cross_attention_optimization_default = "Sub-quadratic"
else: # cuda
    cross_attention_optimization_default ="Scaled-Dot-Product"


options_templates.update(options_section(('sd', "Execution & Models"), {
    "sd_backend": OptionInfo("diffusers" if cmd_opts.use_openvino else "original", "Execution backend", gr.Radio, {"choices": ["original", "diffusers"] }),
    "sd_checkpoint_autoload": OptionInfo(True, "Model autoload on server start"),
    "sd_model_checkpoint": OptionInfo(default_checkpoint, "Base model", gr.Dropdown, lambda: {"choices": list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_model_refiner": OptionInfo('None', "Refiner model", gr.Dropdown, lambda: {"choices": ['None'] + list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_vae": OptionInfo("Automatic", "VAE model", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list),
    "sd_model_dict": OptionInfo('None', "Use baseline data from a different model", gr.Dropdown, lambda: {"choices": ['None'] + list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "stream_load": OptionInfo(False, "Load models using stream loading method"),
    "model_reuse_dict": OptionInfo(False, "When loading models attempt to reuse previous model dictionary", gr.Checkbox, {"visible": False}),
    "prompt_attention": OptionInfo("Full parser", "Prompt attention parser", gr.Radio, {"choices": ["Full parser", "Compel parser", "A1111 parser", "Fixed attention"] }),
    "prompt_mean_norm": OptionInfo(True, "Prompt attention mean normalization"),
    "comma_padding_backtrack": OptionInfo(20, "Prompt padding for long prompts", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1 }),
    "sd_checkpoint_cache": OptionInfo(0, "Number of cached models", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_vae_checkpoint_cache": OptionInfo(0, "Number of cached VAEs", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "sd_disable_ckpt": OptionInfo(False, "Disallow usage of models in ckpt format"),
}))

options_templates.update(options_section(('cuda', "Compute Settings"), {
    "math_sep": OptionInfo("<h2>Execution precision</h2>", "", gr.HTML),
    "precision": OptionInfo("Autocast", "Precision type", gr.Radio, {"choices": ["Autocast", "Full"]}),
    "cuda_dtype": OptionInfo("FP32" if sys.platform == "darwin" or cmd_opts.use_openvino else "BF16" if devices.backend == "ipex" else "FP16", "Device precision type", gr.Radio, {"choices": ["FP32", "FP16", "BF16"]}),
    "no_half": OptionInfo(True if cmd_opts.use_openvino else False, "Use full precision for model (--no-half)", None, None, None),
    "no_half_vae": OptionInfo(True if cmd_opts.use_openvino else False, "Use full precision for VAE (--no-half-vae)"),
    "upcast_sampling": OptionInfo(True if sys.platform == "darwin" else False, "Enable upcast sampling"),
    "upcast_attn": OptionInfo(False, "Enable upcast cross attention layer"),
    "cuda_cast_unet": OptionInfo(False, "Use fixed UNet precision"),
    "disable_nan_check": OptionInfo(True, "Disable NaN check in produced images/latent spaces", gr.Checkbox, {"visible": False}),
    "rollback_vae": OptionInfo(False, "Attempt VAE roll back when produced NaN values"),

    "cross_attention_sep": OptionInfo("<h2>Cross-attention</h2>", "", gr.HTML),
    "cross_attention_optimization": OptionInfo(cross_attention_optimization_default, "Cross-attention optimization method", gr.Radio, lambda: {"choices": shared_items.list_crossattention() }),
    "cross_attention_options": OptionInfo([], "Cross-attention advanced options", gr.CheckboxGroup, {"choices": ['xFormers enable flash Attention', 'SDP disable memory attention']}),
    "sub_quad_sep": OptionInfo("<h3>Sub-quadratic options</h3>", "", gr.HTML),
    "sub_quad_q_chunk_size": OptionInfo(512, "cross-attention query chunk size", gr.Slider, {"minimum": 16, "maximum": 8192, "step": 8}),
    "sub_quad_kv_chunk_size": OptionInfo(512, "cross-attention kv chunk size", gr.Slider, {"minimum": 0, "maximum": 8192, "step": 8}),
    "sub_quad_chunk_threshold": OptionInfo(80, "cross-attention chunking threshold", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),

    "other_sep": OptionInfo("<h2>Execution precision</h2>", "", gr.HTML),
    "opt_channelslast": OptionInfo(False, "Use channels last as torch memory format "),
    "cudnn_benchmark": OptionInfo(False, "Enable full-depth cuDNN benchmark feature"),
    "torch_gc_threshold": OptionInfo(90, "VRAM usage threshold before running Torch GC to clear up VRAM", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),

    "cuda_compile_sep": OptionInfo("<h2>Model Compile</h2>", "", gr.HTML),
    "cuda_compile": OptionInfo(True if cmd_opts.use_openvino else False, "Enable model compile"),
    "cuda_compile_upscaler": OptionInfo(False, "Enable upscaler compile"),
    "cuda_compile_backend": OptionInfo("openvino_fx" if cmd_opts.use_openvino else "none", "Model compile backend", gr.Radio, {"choices": ['none', 'inductor', 'cudagraphs', 'aot_ts_nvfuser', 'hidet', 'ipex', 'openvino_fx']}),
    "cuda_compile_mode": OptionInfo("default", "Model compile mode", gr.Radio, {"choices": ['default', 'reduce-overhead', 'max-autotune']}),
    "cuda_compile_fullgraph": OptionInfo(False, "Model compile fullgraph"),
    "cuda_compile_precompile": OptionInfo(False, "Model compile precompile"),
    "cuda_compile_verbose": OptionInfo(False, "Model compile verbose mode"),
    "cuda_compile_errors": OptionInfo(True, "Model compile suppress errors"),

    "ipex_sep": OptionInfo("<h2>IPEX, DirectML and OpenVINO</h2>", "", gr.HTML),
    "ipex_optimize": OptionInfo(True if devices.backend == "ipex" else False, "Enable IPEX Optimize for Intel GPUs"),
    "ipex_optimize_upscaler": OptionInfo(True if devices.backend == "ipex" else False, "Enable IPEX Optimize for Intel GPUs with Upscalers"),
    "directml_memory_provider": OptionInfo(default_memory_provider, 'DirectML memory stats provider', gr.Radio, {"choices": memory_providers}),
    "directml_catch_nan": OptionInfo(False, "DirectML retry specific operation when NaN is produced if possible. (makes generation slower)"),
    "openvino_disable_model_caching": OptionInfo(False, "OpenVINO disable model caching"),
    "openvino_hetero_gpu": OptionInfo(False, "OpenVINO use Hetero Device for single inference with multiple devices"),
    "openvino_remove_cpu_from_hetero": OptionInfo(False, "OpenVINO remove CPU from Hetero Device"),
    "openvino_remove_igpu_from_hetero": OptionInfo(False, "OpenVINO remove iGPU from Hetero Device"),
}))

options_templates.update(options_section(('advanced', "Inference Settings"), {
    "token_merging_sep": OptionInfo("<h2>Token merging</h2>", "", gr.HTML),
    "token_merging_ratio": OptionInfo(0.0, "Token merging ratio (txt2img)", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}),
    "token_merging_ratio_img2img": OptionInfo(0.0, "Token merging ratio (img2img)", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}),
    "token_merging_ratio_hr": OptionInfo(0.0, "Token merging ratio (hires)", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}),

    "freeu_sep": OptionInfo("<h2>FreeU</h2>", "", gr.HTML),
    "freeu_enabled": OptionInfo(False, "FreeU enabled"),
    "freeu_b1": OptionInfo(1.2, "1st stage backbone factor", gr.Slider, {"minimum": 1.0, "maximum": 2.0, "step": 0.01}),
    "freeu_b2": OptionInfo(1.4, "2nd stage backbone factor", gr.Slider, {"minimum": 1.0, "maximum": 2.0, "step": 0.01}),
    "freeu_s1": OptionInfo(0.9, "1st stage skip factor", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "freeu_s2": OptionInfo(0.2, "2nd stage skip factor", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),

    "hypertile_sep": OptionInfo("<h2>HyperTile</h2>", "", gr.HTML),
    "hypertile_vae_enabled": OptionInfo(False, "HyperTile for VAE enabled", gr.Checkbox, {"visible": False}),
    "hypertile_vae_tile": OptionInfo(128, "HyperTile for VAE tile size", gr.Slider, {"minimum": 128, "maximum": 512, "step": 8, "visible": False}),
    "hypertile_unet_enabled": OptionInfo(False, "HyperTile for UNet enabled"),
    "hypertile_unet_tile": OptionInfo(256, "HyperTile for UNet tile size", gr.Slider, {"minimum": 256, "maximum": 1024, "step": 8}),

    "inference_other_sep": OptionInfo("<h2>Other</h2>", "", gr.HTML),
    "batch_frame_mode": OptionInfo(False, "Process multiple images in batch in parallel"),
    "inference_mode": OptionInfo("no-grad", "Torch inference mode", gr.Radio, {"choices": ["no-grad", "inference-mode", "none"]}),
    "sd_vae_sliced_encode": OptionInfo(False, "VAE Slicing (original)"),
}))

options_templates.update(options_section(('diffusers', "Diffusers Settings"), {
    "diffusers_pipeline": OptionInfo('Autodetect', 'Diffusers pipeline', gr.Dropdown, lambda: {"choices": list(shared_items.get_pipelines()) }),
    "diffusers_move_base": OptionInfo(True, "Move base model to CPU when using refiner"),
    "diffusers_move_unet": OptionInfo(True, "Move base model to CPU when using VAE"),
    "diffusers_move_refiner": OptionInfo(True, "Move refiner model to CPU when not in use"),
    "diffusers_extract_ema": OptionInfo(True, "Use model EMA weights when possible"),
    "diffusers_generator_device": OptionInfo("default", "Generator device", gr.Radio, {"choices": ["default", "cpu"]}),
    "diffusers_model_cpu_offload": OptionInfo(False, "Enable model CPU offload (--medvram)"),
    "diffusers_seq_cpu_offload": OptionInfo(False, "Enable sequential CPU offload (--lowvram)"),
    "diffusers_vae_upcast": OptionInfo("default", "VAE upcasting", gr.Radio, {"choices": ['default', 'true', 'false']}),
    "diffusers_vae_slicing": OptionInfo(True, "Enable VAE slicing"),
    "diffusers_vae_tiling": OptionInfo(False if cmd_opts.use_openvino else True, "Enable VAE tiling"),
    "diffusers_attention_slicing": OptionInfo(False, "Enable attention slicing"),
    "diffusers_model_load_variant": OptionInfo("default", "Diffusers model loading variant", gr.Radio, {"choices": ['default', 'fp32', 'fp16']}),
    "diffusers_vae_load_variant": OptionInfo("default", "Diffusers VAE loading variant", gr.Radio, {"choices": ['default', 'fp32', 'fp16']}),
    "custom_diffusers_pipeline": OptionInfo('hf-internal-testing/diffusers-dummy-pipeline', 'Custom Diffusers pipeline to use'),
    "diffusers_lora_loader": OptionInfo("diffusers" if cmd_opts.use_openvino else "sequential apply", "Diffusers LoRA loading variant", gr.Radio, {"choices": ['diffusers', 'sequential apply', 'merge and apply']}),
    "diffusers_force_zeros": OptionInfo(True, "Force zeros for prompts when empty"),
    "diffusers_aesthetics_score": OptionInfo(False, "Require aesthetics score"),
    "diffusers_force_inpaint": OptionInfo(False, 'Diffusers force inpaint pipeline'),
    "diffusers_pooled": OptionInfo("default", "Diffusers SDXL pooled embeds (experimental)", gr.Radio, {"choices": ['default', 'weighted']}),
}))

options_templates.update(options_section(('system-paths', "System Paths"), {
    "temp_dir": OptionInfo("", "Directory for temporary images; leave empty for default", folder=True),
    "clean_temp_dir_at_start": OptionInfo(True, "Cleanup non-default temporary directory when starting webui"),
    "ckpt_dir": OptionInfo(os.path.join(paths.models_path, 'Stable-diffusion'), "Folder with stable diffusion models", folder=True),
    "diffusers_dir": OptionInfo(os.path.join(paths.models_path, 'Diffusers'), "Folder with Hugggingface models", folder=True),
    "vae_dir": OptionInfo(os.path.join(paths.models_path, 'VAE'), "Folder with VAE files", folder=True),
    "sd_lora": OptionInfo("", "Add LoRA to prompt", gr.Textbox, {"visible": False}),
    "lora_dir": OptionInfo(os.path.join(paths.models_path, 'Lora'), "Folder with LoRA network(s)", folder=True),
    "lyco_dir": OptionInfo(os.path.join(paths.models_path, 'LyCORIS'), "Folder with LyCORIS network(s)", gr.Text, {"visible": False}),
    "styles_dir": OptionInfo(os.path.join(paths.data_path, 'styles.csv'), "File or Folder with user-defined styles", folder=True),
    "embeddings_dir": OptionInfo(os.path.join(paths.models_path, 'embeddings'), "Folder with textual inversion embeddings", folder=True),
    "hypernetwork_dir": OptionInfo(os.path.join(paths.models_path, 'hypernetworks'), "Folder with Hypernetwork models", folder=True),
    "codeformer_models_path": OptionInfo(os.path.join(paths.models_path, 'Codeformer'), "Folder with codeformer models", folder=True),
    "gfpgan_models_path": OptionInfo(os.path.join(paths.models_path, 'GFPGAN'), "Folder with GFPGAN models", folder=True),
    "esrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'ESRGAN'), "Folder with ESRGAN models", folder=True),
    "bsrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'BSRGAN'), "Folder with BSRGAN models", folder=True),
    "realesrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'RealESRGAN'), "Folder with RealESRGAN models", folder=True),
    "scunet_models_path": OptionInfo(os.path.join(paths.models_path, 'SCUNet'), "Folder with SCUNet models", folder=True),
    "swinir_models_path": OptionInfo(os.path.join(paths.models_path, 'SwinIR'), "Folder with SwinIR models", folder=True),
    "ldsr_models_path": OptionInfo(os.path.join(paths.models_path, 'LDSR'), "Folder with LDSR models", folder=True),
    "clip_models_path": OptionInfo(os.path.join(paths.models_path, 'CLIP'), "Folder with CLIP models", folder=True),
}))

options_templates.update(options_section(('saving-images', "Image Options"), {
    "keep_incomplete": OptionInfo(True, "Keep incomplete images"),
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('jpg', 'File format for generated images', gr.Dropdown, {"choices": ["jpg", "png", "webp", "tiff", "jp2"]}),
    "jpeg_quality": OptionInfo(90, "Quality for saved images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "img_max_size_mp": OptionInfo(250, "Maximum image size (MP)", gr.Slider, {"minimum": 100, "maximum": 2000, "step": 1}),
    "webp_lossless": OptionInfo(False, "Use lossless compression for webp images"),
    "save_selected_only": OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "samples_save_zip": OptionInfo(True, "Create zip archive when downloading multiple images"),

    "image_sep_metadata": OptionInfo("<h2>Metadata/Logging</h2>", "", gr.HTML),
    "image_metadata": OptionInfo(True, "Include metadata in saved images"),
    "save_txt": OptionInfo(False, "Create text file next to every image with generation parameters"),
    "save_log_fn": OptionInfo("", "Create JSON log file for each saved image", component_args=hide_dirs),
    "image_watermark_enabled": OptionInfo(False, "Include watermark in saved images"),
    "image_watermark": OptionInfo('', "Image watermark string"),
    "image_sep_grid": OptionInfo("<h2>Grid Options</h2>", "", gr.HTML),
    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('jpg', 'File format for grids', gr.Dropdown, {"choices": ["jpg", "png", "webp", "tiff", "jp2"]}),
    "n_rows": OptionInfo(-1, "Grid row count", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),

    "save_sep_options": OptionInfo("<h2>Intermediate Image Saving</h2>", "", gr.HTML),
    "save_init_img": OptionInfo(False, "Save copy of img2img init images"),
    "save_images_before_highres_fix": OptionInfo(False, "Save copy of image before applying hires"),
    "save_images_before_refiner": OptionInfo(False, "Save copy of image before running refiner"),
    "save_images_before_face_restoration": OptionInfo(False, "Save copy of image before doing face restoration"),
    "save_images_before_color_correction": OptionInfo(False, "Save copy of image before applying color correction"),
    "save_mask": OptionInfo(False, "Save copy of the inpainting greyscale mask"),
    "save_mask_composite": OptionInfo(False, "Save copy of inpainting masked composite"),
}))

options_templates.update(options_section(('saving-paths', "Image Naming & Paths"), {
    "saving_sep_images": OptionInfo("<h2>Save options</h2>", "", gr.HTML),
    "save_images_add_number": OptionInfo(True, "Add number to filename when saving", component_args=hide_dirs),
    "use_original_name_batch": OptionInfo(True, "Use original name during batch process"),
    "use_upscaler_name_as_suffix": OptionInfo(True, "Use upscaler as suffix", gr.Checkbox, {"visible": False}),
    "samples_filename_pattern": OptionInfo("[seq]-[model_name]-[prompt_words]", "Images filename pattern", component_args=hide_dirs),
    "directories_max_prompt_words": OptionInfo(8, "Max words for [prompt_words] pattern", gr.Slider, {"minimum": 1, "maximum": 99, "step": 1, **hide_dirs}),
    "use_save_to_dirs_for_ui": OptionInfo(False, "Save images to a subdirectory when using Save button", gr.Checkbox, {"visible": False}),
    "save_to_dirs": OptionInfo(False, "Save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs),

    "outdir_sep_dirs": OptionInfo("<h2>Directories</h2>", "", gr.HTML),
    "outdir_samples": OptionInfo("", "Output directory for images", component_args=hide_dirs, folder=True),
    "outdir_txt2img_samples": OptionInfo("outputs/text", 'Output directory for txt2img images', component_args=hide_dirs, folder=True),
    "outdir_img2img_samples": OptionInfo("outputs/image", 'Output directory for img2img images', component_args=hide_dirs, folder=True),
    "outdir_extras_samples": OptionInfo("outputs/extras", 'Output directory for images from extras tab', component_args=hide_dirs, folder=True),
    "outdir_save": OptionInfo("outputs/save", "Directory for saving images using the Save button", component_args=hide_dirs, folder=True),
    "outdir_init_images": OptionInfo("outputs/init-images", "Directory for saving init images when using img2img", component_args=hide_dirs, folder=True),

    "outdir_sep_grids": OptionInfo("<h2>Grids</h2>", "", gr.HTML),
    "grid_extended_filename": OptionInfo(True, "Add extended info (seed, prompt) to filename when saving grid", gr.Checkbox, {"visible": False}),
    "grid_save_to_dirs": OptionInfo(False, "Save grids to a subdirectory", gr.Checkbox, {"visible": False}),
    "outdir_grids": OptionInfo("", "Output directory for grids", component_args=hide_dirs, folder=True),
    "outdir_txt2img_grids": OptionInfo("outputs/grids", 'Output directory for txt2img grids', component_args=hide_dirs, folder=True),
    "outdir_img2img_grids": OptionInfo("outputs/grids", 'Output directory for img2img grids', component_args=hide_dirs, folder=True),
}))

options_templates.update(options_section(('ui', "User Interface"), {
    "motd": OptionInfo(True, "Show MOTD"),
    "gradio_theme": OptionInfo("black-teal", "UI theme", gr.Dropdown, lambda: {"choices": theme.list_themes()}, refresh=theme.refresh_themes),
    "theme_style": OptionInfo("Auto", "Theme mode", gr.Radio, {"choices": ["Auto", "Dark", "Light"]}),
    "tooltips": OptionInfo("UI Tooltips", "UI tooltips", gr.Radio, {"choices": ["None", "Browser default", "UI tooltips"], "visible": False}),
    "gallery_height": OptionInfo("", "Gallery height", gr.Textbox),
    "compact_view": OptionInfo(False, "Compact view"),
    "return_grid": OptionInfo(True, "Show grid in results"),
    "return_mask": OptionInfo(False, "For inpainting, include the greyscale mask in results"),
    "return_mask_composite": OptionInfo(False, "For inpainting, include masked composite in results"),
    "disable_weights_auto_swap": OptionInfo(True, "Do not change selected model when reading generation parameters"),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
    "font": OptionInfo("", "Font for image grids that have text"),
    "keyedit_precision_attention": OptionInfo(0.1, "Ctrl+up/down precision when editing (attention:1.1)", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001, "visible": False}),
    "keyedit_precision_extra": OptionInfo(0.05, "Ctrl+up/down precision when editing <extra networks:0.9>", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001, "visible": False}),
    "keyedit_delimiters": OptionInfo(".,\/!?%^*;:{}=`~()", "Ctrl+up/down word delimiters", gr.Textbox, { "visible": False }), # pylint: disable=anomalous-backslash-in-string
    "quicksettings_list": OptionInfo(["sd_model_checkpoint"] if backend == Backend.ORIGINAL else ["sd_model_checkpoint", "sd_model_refiner"], "Quicksettings list", ui_components.DropdownMulti, lambda: {"choices": list(opts.data_labels.keys())}),
    "ui_scripts_reorder": OptionInfo("", "UI scripts order", gr.Textbox, { "visible": False }),
}))

options_templates.update(options_section(('live-preview', "Live Previews"), {
    "show_progressbar": OptionInfo(True, "Show progressbar", gr.Checkbox, {"visible": False}),
    "live_previews_enable": OptionInfo(True, "Show live previews of the created image", gr.Checkbox, {"visible": False}),
    "show_progress_grid": OptionInfo(True, "Show previews of all images generated in a batch as a grid", gr.Checkbox, {"visible": False}),
    "notification_audio_enable": OptionInfo(False, "Play a sound when images are finished generating"),
    "notification_audio_path": OptionInfo("html/notification.mp3","Path to notification sound", component_args=hide_dirs, folder=True),
    "show_progress_every_n_steps": OptionInfo(1, "Live preview display period", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "show_progress_type": OptionInfo("Approximate", "Live preview method", gr.Radio, {"choices": ["Simple", "Approximate", "TAESD", "Full VAE"]}),
    "live_preview_content": OptionInfo("Combined", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"], "visible": False}),
    "live_preview_refresh_period": OptionInfo(500, "Progress update period", gr.Slider, {"minimum": 0, "maximum": 5000, "step": 25}),
    "logmonitor_show": OptionInfo(True, "Show log view"),
    "logmonitor_refresh_period": OptionInfo(5000, "Log view update period", gr.Slider, {"minimum": 0, "maximum": 30000, "step": 25}),
}))

options_templates.update(options_section(('sampler-params', "Sampler Settings"), {
    "show_samplers": OptionInfo([], "Show samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in list_samplers()]}),
    'eta_noise_seed_delta': OptionInfo(0, "Noise seed delta (eta)", gr.Number, {"precision": 0}),
    "scheduler_eta": OptionInfo(1.0, "Noise multiplier (eta)", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "schedulers_solver_order": OptionInfo(2, "Solver order (where applicable)", gr.Slider, {"minimum": 1, "maximum": 5, "step": 1}),

    # managed from ui.py for backend original
    "schedulers_brownian_noise": OptionInfo(True, "Use Brownian noise", gr.Checkbox, {"visible": False}),
    "schedulers_discard_penultimate": OptionInfo(True, "Discard penultimate sigma", gr.Checkbox, {"visible": False}),
    "schedulers_sigma": OptionInfo("default", "Sigma algorithm", gr.Radio, {"choices": ['default', 'karras', 'exponential', 'polyexponential'], "visible": False}),
    "schedulers_use_karras": OptionInfo(True, "Use Karras sigmas", gr.Checkbox, {"visible": False}),
    "schedulers_use_thresholding": OptionInfo(False, "Use dynamic thresholding", gr.Checkbox, {"visible": False}),
    "schedulers_use_loworder": OptionInfo(True, "Use simplified solvers in final steps", gr.Checkbox, {"visible": False}),
    "schedulers_prediction_type": OptionInfo("default", "Override model prediction type", gr.Radio, {"choices": ['default', 'epsilon', 'sample', 'v_prediction'], "visible": False}),

    # managed from ui.py for backend diffusers
    "schedulers_sep_diffusers": OptionInfo("<h2>Diffusers specific config</h2>", "", gr.HTML),
    "schedulers_dpm_solver": OptionInfo("sde-dpmsolver++", "DPM solver algorithm", gr.Radio, {"choices": ['dpmsolver', 'dpmsolver++', 'sde-dpmsolver', 'sde-dpmsolver++']}),
    "schedulers_beta_schedule": OptionInfo("default", "Beta schedule", gr.Radio, {"choices": ['default', 'linear', 'scaled_linear', 'squaredcos_cap_v2']}),
    'schedulers_beta_start': OptionInfo(0, "Beta start", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.00001}),
    'schedulers_beta_end': OptionInfo(0, "Beta end", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.00001}),

    # managed from ui.py for backend original k-diffusion
    "schedulers_sep_kdiffusers": OptionInfo("<h2>K-Diffusion specific config</h2>", "", gr.HTML),
    "always_batch_cond_uncond": OptionInfo(False, "Disable conditional batching enabled on low memory systems"),
    "enable_quantization": OptionInfo(True, "Enable quantization for sharper and cleaner results"),
    's_churn': OptionInfo(0.0, "Sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_min_uncond': OptionInfo(0.0, "Sigma negative guidance minimum ", gr.Slider, {"minimum": 0.0, "maximum": 4.0, "step": 0.01}),
    's_tmin':  OptionInfo(0.0, "Sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise': OptionInfo(1.0, "Sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_min':  OptionInfo(0.0, "Sigma min",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_max':  OptionInfo(0.0, "Sigma max",  gr.Slider, {"minimum": 0.0, "maximum": 100.0, "step": 1.0}),
    "schedulers_sep_compvis": OptionInfo("<h2>CompVis specific config</h2>", "", gr.HTML),
    'uni_pc_variant': OptionInfo("bh1", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"]}),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"]}),
    "ddim_discretize": OptionInfo('uniform', "DDIM discretize img2img", gr.Radio, {"choices": ['uniform', 'quad']}),
    # TODO pad_cond_uncond implementation missing
    "pad_cond_uncond": OptionInfo(True, "Pad prompt and negative prompt to be same length", gr.Checkbox, {"visible": False}),
    # TODO batch_cond-uncond implementation missing
    "batch_cond_uncond": OptionInfo(True, "Do conditional and unconditional denoising in one batch", gr.Checkbox, {"visible": False}),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Enable additional postprocessing operations", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", ui_components.DropdownMulti, lambda: {"choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    "postprocessing_sep_img2img": OptionInfo("<h2>Img2Img & Inpainting</h2>", "", gr.HTML),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to match original colors"),
    "img2img_fix_steps": OptionInfo(False, "For image processing do exact number of steps as specified"),
    "img2img_background_color": OptionInfo("#ffffff", "Image transparent color fill", ui_components.FormColorPicker, {}),
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for image processing", gr.Slider, {"minimum": 0.1, "maximum": 1.5, "step": 0.01}),
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 8, "step": 1, "visible": False}),

    "postprocessing_sep_face_restoration": OptionInfo("<h2>Face Restoration</h2>", "", gr.HTML),
    "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.2, "CodeFormer weight parameter", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),

    "postprocessing_sep_upscalers": OptionInfo("<h2>Upscaling</h2>", "", gr.HTML),
    "upscaler_unload": OptionInfo(False, "Unload upscaler after processing"),
    # 'upscaling_max_images_in_cache': OptionInfo(5, "Maximum number of images in upscaling cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1, "visible": False}),
    "upscaler_for_img2img": OptionInfo("None", "Default upscaler for image resize operations", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
    "upscaler_tile_size": OptionInfo(192, "Upscaler tile size", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "upscaler_tile_overlap": OptionInfo(8, "Upscaler tile overlap", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}),
}))

options_templates.update(options_section(('training', "Training"), {
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training if possible"),
    "pin_memory": OptionInfo(True, "Pin training dataset to memory"),
    "save_optimizer_state": OptionInfo(False, "Save resumable optimizer state when training"),
    "save_training_settings_to_txt": OptionInfo(True, "Save training settings to a text file on training start"),
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex"),
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string"),
    "embeddings_templates_dir": OptionInfo(os.path.join(paths.script_path, 'train', 'templates'), "Embeddings train templates directory", folder=True),
    "training_image_repeats_per_epoch": OptionInfo(1, "Image repeats per epoch", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "training_write_csv_every": OptionInfo(0, "Save loss CSV file every n steps"),
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
    "interrogate_clip_dict_limit": OptionInfo(2048, "CLIP: maximum number of lines in text file", gr.Slider, { "visible": False }),
    "interrogate_clip_skip_categories": OptionInfo(["artists", "movements", "flavors"], "Interrogate: skip categories", gr.CheckboxGroup, lambda: {"choices": modules.interrogate.category_types()}, refresh=modules.interrogate.category_types),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.65, "Interrogate: deepbooru score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(False, "Interrogate: deepbooru sort alphabetically"),
    "deepbooru_use_spaces": OptionInfo(False, "Use spaces for tags in deepbooru"),
    "deepbooru_escape": OptionInfo(True, "Escape brackets in deepbooru"),
    "deepbooru_filter_tags": OptionInfo("", "Filter out tags from deepbooru output"),
}))

options_templates.update(options_section(('extra_networks', "Extra Networks"), {
    "extra_networks_sep1": OptionInfo("<h2>Extra networks UI</h2>", "", gr.HTML),
    "extra_networks": OptionInfo(["All"], "Extra networks", ui_components.DropdownMulti, lambda: {"choices": ['All'] + [en.title for en in extra_networks]}),
    "extra_networks_card_cover": OptionInfo("sidebar", "UI position", gr.Radio, {"choices": ["cover", "inline", "sidebar"]}),
    "extra_networks_height": OptionInfo(53, "UI height (%)", gr.Slider, {"minimum": 10, "maximum": 100, "step": 1}),
    "extra_networks_sidebar_width": OptionInfo(35, "UI sidebar width (%)", gr.Slider, {"minimum": 10, "maximum": 80, "step": 1}),
    "extra_networks_card_size": OptionInfo(160, "UI card size (px)", gr.Slider, {"minimum": 20, "maximum": 2000, "step": 1}),
    "extra_networks_card_square": OptionInfo(True, "UI disable variable aspect ratio"),
    "extra_networks_card_fit": OptionInfo("cover", "UI image contain method", gr.Radio, {"choices": ["contain", "cover", "fill"], "visible": False}),

    "extra_networks_sep2": OptionInfo("<h2>Extra networks general</h2>", "", gr.HTML),
    "extra_network_skip_indexing": OptionInfo(False, "Do not automatically build extra network pages", gr.Checkbox),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Default multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),

    "extra_networks_sep3": OptionInfo("<h2>Extra networks settings</h2>", "", gr.HTML),
    "extra_networks_styles": OptionInfo(True, "Show built-in styles"),
    "lora_preferred_name": OptionInfo("filename", "LoRA preffered name", gr.Radio, {"choices": ["filename", "alias"]}),
    "lora_add_hashes_to_infotext": OptionInfo(True, "LoRA add hash info"),
    "lora_in_memory_limit": OptionInfo(0, "LoRA memory cache", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "lora_functional": OptionInfo(False, "Use Kohya method for handling multiple LoRA", gr.Checkbox, { "visible": False }),

    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, { "choices": ["None"], "visible": False }),
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
    filename = None
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
        if oldval is None:
            oldval = self.data_labels[key].default
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
        return data_label.default if data_label is not None else None

    def save(self, filename=None, silent=False):
        if filename is None:
            filename = self.filename
        if cmd_opts.freeze:
            log.warning(f'Settings saving is disabled: {filename}')
            return
        try:
            # output = json.dumps(self.data, indent=2)
            diff = {}
            unused_settings = []
            for k, v in self.data.items():
                if k in self.data_labels:
                    if type(v) is list:
                        diff[k] = v
                    if self.data_labels[k].default != v:
                        diff[k] = v
                else:
                    if k not in compatibility_opts:
                        unused_settings.append(k)
                    diff[k] = v
            writefile(diff, filename, silent=silent)
            if len(unused_settings) > 0:
                log.debug(f"Unused settings: {unused_settings}")
        except Exception as e:
            log.error(f'Saving settings failed: {filename} {e}')

    def same_type(self, x, y):
        if x is None or y is None:
            return True
        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))
        return type_x == type_y

    def load(self, filename=None):
        if filename is None:
            filename = self.filename
        if not os.path.isfile(filename):
            log.debug(f'Created default config: {filename}')
            self.save(filename)
            return
        self.data = readfile(filename)
        if self.data.get('quicksettings') is not None and self.data.get('quicksettings_list') is None:
            self.data['quicksettings_list'] = [i.strip() for i in self.data.get('quicksettings').split(',')]
        unknown_settings = []
        for k, v in self.data.items():
            info = self.data_labels.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                log.error(f"Error: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})")
            if info is None and k not in compatibility_opts:
                unknown_settings.append(k)
        if len(unknown_settings) > 0:
            log.debug(f"Unknown settings: {unknown_settings}")

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func
        if call:
            func()

    def dumpjson(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        metadata = {
            k: {
                "is_stored": k in self.data and self.data[k] != self.data_labels[k].default, # pylint: disable=unnecessary-dict-index-lookup
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
    backend = Backend.DIFFUSERS if cmd_opts.use_openvino or opts.data.get('sd_backend', 'original') == 'diffusers' else Backend.ORIGINAL
else:
    backend = Backend.DIFFUSERS if cmd_opts.use_openvino or cmd_opts.backend.lower() == 'diffusers' else Backend.ORIGINAL
opts.data['sd_backend'] = 'diffusers' if backend == Backend.DIFFUSERS else 'original'
if cmd_opts.use_xformers:
    opts.data['cross_attention_optimization'] = 'xFormers'
opts.data['uni_pc_lower_order_final'] = opts.schedulers_use_loworder # compatibility
opts.data['uni_pc_order'] = opts.schedulers_solver_order # compatibility
log.info(f'Engine: backend={backend} compute={devices.backend} mode={devices.inference_context.__name__} device={devices.get_optimal_device_name()} cross-optimization="{opts.cross_attention_optimization}"')
log.info(f'Device: {print_dict(devices.get_gpu_info())}')

prompt_styles = modules.styles.StyleDatabase(opts)
cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or (cmd_opts.server_name or False)) and not cmd_opts.insecure
devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])
device = devices.device
batch_cond_uncond = opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram
mem_mon = modules.memmon.MemUsageMonitor("MemMon", devices.device)
if devices.backend == "directml":
    directml_do_hijack()


class TotalTQDM: # compatibility with previous global-tqdm
    # import tqdm
    def __init__(self):
        pass
    def reset(self):
        pass
    def update(self):
        pass
    def updateTotal(self, new_total):
        pass
    def clear(self):
        pass
total_tqdm = TotalTQDM()


def restart_server(restart=True):
    if demo is None:
        return
    log.warning('Server shutdown requested')
    try:
        sys.tracebacklimit = 0
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stdout(stderr):
            demo.server.wants_restart = restart
            demo.server.should_exit = True
            demo.server.force_exit = True
            demo.close(verbose=False)
            demo.server.close()
            demo.fns = []
        time.sleep(1)
        sys.tracebacklimit = 100
        # os._exit(0)
    except (Exception, BaseException) as e:
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


def req(url_addr, headers = None, **kwargs):
    if headers is None:
        headers = { 'Content-type': 'application/json' }
    try:
        res = requests.get(url_addr, timeout=30, headers=headers, verify=False, allow_redirects=True, **kwargs)
    except Exception as e:
        log.error(f'HTTP request error: url={url_addr} {e}')
        res = { 'status_code': 500, 'text': f'HTTP request error: url={url_addr} {e}' }
        res = SimpleNamespace(**res)
    return res

class Shared(sys.modules[__name__].__class__): # this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than at program startup.
    @property
    def sd_model(self):
        # log.debug(f'Access shared.sd_model: {sys._getframe().f_back.f_code.co_name}') # pylint: disable=protected-access
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
        return Backend.ORIGINAL if not cmd_opts.use_openvino and opts.data['sd_backend'] == 'original' else Backend.DIFFUSERS

    @property
    def sd_model_type(self):
        try:
            import modules.sd_models # pylint: disable=W0621
            if modules.sd_models.model_data.sd_model is None:
                model_type = 'none'
                return model_type
            if backend == Backend.ORIGINAL:
                model_type = 'ldm'
            elif "StableDiffusionXL" in self.sd_model.__class__.__name__:
                model_type = 'sdxl'
            elif "StableDiffusion" in self.sd_model.__class__.__name__:
                model_type = 'sd'
            elif "Kandinsky" in self.sd_model.__class__.__name__:
                model_type = 'kandinsky'
            else:
                model_type = self.sd_model.__class__.__name__
        except Exception:
            model_type = 'unknown'
        return model_type

    @property
    def sd_refiner_type(self):
        try:
            import modules.sd_models # pylint: disable=W0621
            if modules.sd_models.model_data.sd_refiner is None:
                model_type = 'none'
                return model_type
            if backend == Backend.ORIGINAL:
                model_type = 'ldm'
            elif "StableDiffusionXL" in self.sd_refiner.__class__.__name__:
                model_type = 'sdxl'
            elif "StableDiffusion" in self.sd_refiner.__class__.__name__:
                model_type = 'sd'
            elif "Kandinsky" in self.sd_refiner.__class__.__name__:
                model_type = 'kandinsky'
            else:
                model_type = self.sd_refiner.__class__.__name__
        except Exception:
            model_type = 'unknown'
        return model_type

sd_model = None
sd_refiner = None
sd_model_type = ''
sd_refiner_type = ''
compiled_model_state = None
sys.modules[__name__].__class__ = Shared
