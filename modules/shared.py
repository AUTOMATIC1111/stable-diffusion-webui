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
import orjson
import diffusers
from rich.console import Console
from modules import errors, shared_items, shared_state, cmd_args, theme
from modules.paths import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir # pylint: disable=W0611
from modules.dml import memory_providers, default_memory_provider, directml_do_hijack
from modules.onnx_impl import initialize as initialize_onnx
from modules.onnx_impl.execution_providers import available_execution_providers, get_default_execution_provider
import modules.interrogate
import modules.memmon
import modules.styles
import modules.devices as devices # pylint: disable=R0402
import modules.paths as paths
from installer import print_dict
from installer import log as central_logger # pylint: disable=E0611


errors.install([gr])
demo: gr.Blocks = None
api = None
log = central_logger
progress_print_out = sys.stdout
parser = cmd_args.parser
url = 'https://github.com/vladmandic/automatic'
cmd_opts, _ = parser.parse_known_args()
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
xformers_available = False
locking_available = True
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
    "Latent Nearest": {"mode": "nearest", "antialias": False},
    "Latent Nearest-exact": {"mode": "nearest-exact", "antialias": False},
    "Latent Area": {"mode": "area", "antialias": False},
    "Latent Bilinear": {"mode": "bilinear", "antialias": False},
    "Latent Bicubic": {"mode": "bicubic", "antialias": False},
    "Latent Bilinear antialias": {"mode": "bilinear", "antialias": True},
    "Latent Bicubic antialias": {"mode": "bicubic", "antialias": True},
    # "Latent Linear": {"mode": "linear", "antialias": False}, # not supported for latents with channels=4
    # "Latent Trilinear": {"mode": "trilinear", "antialias": False}, # not supported for latents with channels=4
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
resize_modes = ["None", "Fixed", "Crop", "Fill", "Latent"]
compatibility_opts = ['clip_skip', 'uni_pc_lower_order_final', 'uni_pc_order']
console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
dir_timestamps = {}
dir_cache = {}


class Backend(Enum):
    ORIGINAL = 1
    DIFFUSERS = 2


state = shared_state.State()
if not hasattr(cmd_opts, "use_openvino"):
    cmd_opts.use_openvino = False


def readfile(filename, silent=False, lock=False):
    global locking_available # pylint: disable=global-statement
    data = {}
    lock_file = None
    locked = False
    if lock and locking_available:
        try:
            lock_file = fasteners.InterProcessReaderWriterLock(f"{filename}.lock")
            lock_file.logger.disabled = True
            locked = lock_file.acquire_read_lock(blocking=True, timeout=3)
        except Exception as e:
            lock_file = None
            locking_available = False
            log.error(f'File read lock: file="{filename}" {e}')
            locked = False
    try:
        # if not os.path.exists(filename):
        #    return {}
        t0 = time.time()
        with open(filename, "rb") as file:
            b = file.read()
            data = orjson.loads(b) # pylint: disable=no-member
        # if type(data) is str:
        #    data = json.loads(data)
        t1 = time.time()
        if not silent:
            log.debug(f'Read: file="{filename}" json={len(data)} bytes={os.path.getsize(filename)} time={t1-t0:.3f}')
    except Exception as e:
        if not silent:
            log.error(f'Reading failed: {filename} {e}')
    try:
        if locking_available and lock_file is not None:
            lock_file.release_read_lock()
        if locked and os.path.exists(f"{filename}.lock"):
            os.remove(f"{filename}.lock")
    except Exception:
        locking_available = False
    return data


def writefile(data, filename, mode='w', silent=False, atomic=False):
    import tempfile
    global locking_available # pylint: disable=global-statement
    lock_file = None
    locked = False

    def default(obj):
        log.error(f'Saving: file="{filename}" not a valid object: {obj}')
        return str(obj)

    try:
        t0 = time.time()
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
    except Exception as e:
        log.error(f'Saving failed: file="{filename}" {e}')
        return
    try:
        if locking_available:
            lock_file = fasteners.InterProcessReaderWriterLock(f"{filename}.lock") if locking_available else None
            lock_file.logger.disabled = True
            locked = lock_file.acquire_write_lock(blocking=True, timeout=3) if lock_file is not None else False
    except Exception as e:
        locking_available = False
        lock_file = None
        log.error(f'File write lock: file="{filename}" {e}')
        locked = False
    try:
        if atomic:
            with tempfile.NamedTemporaryFile(mode=mode, encoding="utf8", delete=False, dir=os.path.dirname(filename)) as f:
                f.write(output)
                f.flush()
                os.fsync(f.fileno())
                os.replace(f.name, filename)
        else:
            with open(filename, mode=mode, encoding="utf8") as file:
                file.write(output)
        t1 = time.time()
        if not silent:
            log.debug(f'Save: file="{filename}" json={len(data)} bytes={len(output)} time={t1-t0:.3f}')
    except Exception as e:
        log.error(f'Saving failed: file="{filename}" {e}')
    try:
        if locking_available and lock_file is not None:
            lock_file.release_write_lock()
        if locked and os.path.exists(f"{filename}.lock"):
            os.remove(f"{filename}.lock")
    except Exception:
        locking_available = False


# early select backend
default_backend = 'diffusers'
early_opts = readfile(cmd_opts.config, silent=True)
early_backend = early_opts.get('sd_backend', default_backend)
backend = Backend.DIFFUSERS if early_backend.lower() == 'diffusers' else Backend.ORIGINAL
if cmd_opts.backend is not None: # override with args
    backend = Backend.DIFFUSERS if cmd_opts.backend.lower() == 'diffusers' else Backend.ORIGINAL
if cmd_opts.use_openvino: # override for openvino
    backend = Backend.DIFFUSERS
    from modules.intel.openvino import get_device_list as get_openvino_device_list # pylint: disable=ungrouped-imports


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


def refresh_upscalers():
    import modules.modelloader # pylint: disable=W0621
    modules.modelloader.load_upscalers()


def list_samplers():
    import modules.sd_samplers # pylint: disable=W0621
    modules.sd_samplers.set_samplers()
    return modules.sd_samplers.all_samplers


def temp_disable_extensions():
    disable_safe = ['sd-webui-controlnet', 'multidiffusion-upscaler-for-automatic1111', 'a1111-sd-webui-lycoris', 'sd-webui-agent-scheduler', 'clip-interrogator-ext', 'stable-diffusion-webui-rembg', 'sd-extension-chainner', 'stable-diffusion-webui-images-browser']
    disable_diffusers = ['sd-webui-controlnet', 'multidiffusion-upscaler-for-automatic1111', 'a1111-sd-webui-lycoris', 'sd-webui-animatediff']
    disable_original = []
    disabled = []
    if cmd_opts.safe:
        for ext in disable_safe:
            if ext not in opts.disabled_extensions:
                disabled.append(ext)
    if backend == Backend.DIFFUSERS:
        for ext in disable_diffusers:
            if ext not in opts.disabled_extensions:
                disabled.append(ext)
    if backend == Backend.ORIGINAL:
        for ext in disable_original:
            if ext not in opts.disabled_extensions:
                disabled.append(ext)
    cmd_opts.controlnet_loglevel = 'WARNING'
    return disabled


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
    "sd_backend": OptionInfo(default_backend, "Execution backend", gr.Radio, {"choices": ["original", "diffusers"] }),
    "sd_model_checkpoint": OptionInfo(default_checkpoint, "Base model", gr.Dropdown, lambda: {"choices": list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_model_refiner": OptionInfo('None', "Refiner model", gr.Dropdown, lambda: {"choices": ['None'] + list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "sd_vae": OptionInfo("Automatic", "VAE model", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list),
    "sd_checkpoint_autoload": OptionInfo(True, "Model autoload on start"),
    "sd_model_dict": OptionInfo('None', "Use separate base dict", gr.Dropdown, lambda: {"choices": ['None'] + list_checkpoint_tiles()}, refresh=refresh_checkpoints),
    "stream_load": OptionInfo(False, "Load models using stream loading method", gr.Checkbox, {"visible": backend == Backend.ORIGINAL }),
    "model_reuse_dict": OptionInfo(False, "Reuse loaded model dictionary", gr.Checkbox, {"visible": False}),
    "prompt_attention": OptionInfo("Full parser", "Prompt attention parser", gr.Radio, {"choices": ["Full parser", "Compel parser", "A1111 parser", "Fixed attention"] }),
    "prompt_mean_norm": OptionInfo(True, "Prompt attention normalization", gr.Checkbox, {"visible": backend == Backend.ORIGINAL }),
    "comma_padding_backtrack": OptionInfo(20, "Prompt padding", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1, "visible": backend == Backend.ORIGINAL }),
    "sd_checkpoint_cache": OptionInfo(0, "Cached models", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1, "visible": backend == Backend.ORIGINAL }),
    "sd_vae_checkpoint_cache": OptionInfo(0, "Cached VAEs", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1, "visible": False}),
    "sd_disable_ckpt": OptionInfo(False, "Disallow models in ckpt format"),
}))

options_templates.update(options_section(('cuda', "Compute Settings"), {
    "math_sep": OptionInfo("<h2>Execution precision</h2>", "", gr.HTML),
    "precision": OptionInfo("Autocast", "Precision type", gr.Radio, {"choices": ["Autocast", "Full"]}),
    "cuda_dtype": OptionInfo("FP32" if sys.platform == "darwin" or cmd_opts.use_openvino else "BF16" if devices.backend == "ipex" else "FP16", "Device precision type", gr.Radio, {"choices": ["FP32", "FP16", "BF16"]}),
    "no_half": OptionInfo(False if not cmd_opts.use_openvino else True, "Full precision for model (--no-half)", None, None, None),
    "no_half_vae": OptionInfo(False if not cmd_opts.use_openvino else True, "Full precision for VAE (--no-half-vae)"),
    "upcast_sampling": OptionInfo(False if sys.platform != "darwin" else True, "Upcast sampling"),
    "upcast_attn": OptionInfo(False, "Upcast attention layer"),
    "cuda_cast_unet": OptionInfo(False, "Fixed UNet precision"),
    "disable_nan_check": OptionInfo(True, "Disable NaN check", gr.Checkbox, {"visible": False}),
    "nan_skip": OptionInfo(False, "Skip Generation if NaN found in latents", gr.Checkbox, {"visible": True}),
    "rollback_vae": OptionInfo(False, "Attempt VAE roll back for NaN values"),

    "cross_attention_sep": OptionInfo("<h2>Attention</h2>", "", gr.HTML),
    "cross_attention_optimization": OptionInfo(cross_attention_optimization_default, "Attention optimization method", gr.Radio, lambda: {"choices": shared_items.list_crossattention() }),
    "cross_attention_options": OptionInfo([], "Attention advanced options", gr.CheckboxGroup, {"choices": ['xFormers enable flash Attention', 'SDP disable memory attention']}),
    "sub_quad_sep": OptionInfo("<h3>Sub-quadratic options</h3>", "", gr.HTML),
    "sub_quad_q_chunk_size": OptionInfo(512, "Attention query chunk size", gr.Slider, {"minimum": 16, "maximum": 8192, "step": 8}),
    "sub_quad_kv_chunk_size": OptionInfo(512, "Attention kv chunk size", gr.Slider, {"minimum": 0, "maximum": 8192, "step": 8}),
    "sub_quad_chunk_threshold": OptionInfo(80, "Attention chunking threshold", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),

    "other_sep": OptionInfo("<h2>Execution precision</h2>", "", gr.HTML),
    "opt_channelslast": OptionInfo(False, "Use channels last "),
    "cudnn_benchmark": OptionInfo(False, "Full-depth cuDNN benchmark feature"),
    "diffusers_fuse_projections": OptionInfo(False, "Fused projections"),
    "torch_gc_threshold": OptionInfo(80, "Memory usage threshold for GC", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),

    "cuda_compile_sep": OptionInfo("<h2>Model Compile</h2>", "", gr.HTML),
    "cuda_compile": OptionInfo([] if not cmd_opts.use_openvino else ["Model", "VAE", "Upscaler"], "Compile Model", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder", "Upscaler"]}),
    "cuda_compile_backend": OptionInfo("none" if not cmd_opts.use_openvino else "openvino_fx", "Model compile backend", gr.Radio, {"choices": ['none', 'inductor', 'cudagraphs', 'aot_ts_nvfuser', 'hidet', 'ipex', 'openvino_fx', 'stable-fast', 'olive-ai']}),
    "cuda_compile_mode": OptionInfo("default", "Model compile mode", gr.Radio, {"choices": ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']}),
    "cuda_compile_fullgraph": OptionInfo(False, "Model compile fullgraph"),
    "cuda_compile_precompile": OptionInfo(False, "Model compile precompile"),
    "cuda_compile_verbose": OptionInfo(False, "Model compile verbose mode"),
    "cuda_compile_errors": OptionInfo(True, "Model compile suppress errors"),
    "diffusers_quantization": OptionInfo(False, "Dynamic quantization with TorchAO"),
    "nncf_compress_weights": OptionInfo([], "Compress Model weights with NNCF", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder"], "visible": backend == Backend.DIFFUSERS}),

    "ipex_sep": OptionInfo("<h2>IPEX</h2>", "", gr.HTML, {"visible": devices.backend == "ipex"}),
    "ipex_optimize": OptionInfo(["Model", "VAE", "Text Encoder", "Upscaler"] if devices.backend == "ipex" else [], "IPEX Optimize for Intel GPUs", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder", "Upscaler"], "visible": devices.backend == "ipex"}),

    "openvino_sep": OptionInfo("<h2>OpenVINO</h2>", "", gr.HTML, {"visible": cmd_opts.use_openvino}),
    "openvino_devices": OptionInfo([], "OpenVINO devices to use", gr.CheckboxGroup, {"choices": get_openvino_device_list() if cmd_opts.use_openvino else [], "visible": cmd_opts.use_openvino}),
    "nncf_quantize": OptionInfo([], "OpenVINO Quantize Models with NNCF", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder"], "visible": cmd_opts.use_openvino}),
    "nncf_quant_mode": OptionInfo("INT8", "OpenVINO quantization mode for NNCF", gr.Radio, {"choices": ['INT8', 'FP8_E4M3', 'FP8_E5M2'], "visible": cmd_opts.use_openvino}),
    "nncf_compress_weights_mode": OptionInfo("INT8", "OpenVINO compress mode for NNCF", gr.Radio, {"choices": ['INT8', 'INT8_SYM', 'INT4_ASYM', 'INT4_SYM', 'NF4'], "visible": cmd_opts.use_openvino}),
    "nncf_compress_weights_raito": OptionInfo(1.0, "OpenVINO compress ratio for NNCF", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01, "visible": cmd_opts.use_openvino}),
    "openvino_disable_model_caching": OptionInfo(False, "OpenVINO disable model caching", gr.Checkbox, {"visible": cmd_opts.use_openvino}),

    "directml_sep": OptionInfo("<h2>DirectML</h2>", "", gr.HTML, {"visible": devices.backend == "directml"}),
    "directml_memory_provider": OptionInfo(default_memory_provider, 'DirectML memory stats provider', gr.Radio, {"choices": memory_providers, "visible": devices.backend == "directml"}),
    "directml_catch_nan": OptionInfo(False, "DirectML retry ops for NaN", gr.Checkbox, {"visible": devices.backend == "directml"}),

    "olive_sep": OptionInfo("<h2>Olive</h2>", "", gr.HTML),
    "olive_float16": OptionInfo(True, 'Olive use FP16 on optimization (will use FP32 if unchecked)'),
    "olive_vae_encoder_float32": OptionInfo(False, 'Olive force FP32 for VAE Encoder (if Img2Img generates NaN, enable this option and remove previously optimized model)'),
    "olive_static_dims": OptionInfo(True, 'Olive use static dimensions (make inference faster with OrtTransformersOptimization)'),
    "olive_cache_optimized": OptionInfo(True, 'Olive cache optimized models'),
}))

options_templates.update(options_section(('advanced', "Inference Settings"), {
    "token_merging_sep": OptionInfo("<h2>Token merging</h2>", "", gr.HTML),
    "token_merging_ratio": OptionInfo(0.0, "Token merging ratio for txt2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}),
    "token_merging_ratio_img2img": OptionInfo(0.0, "Token merging ratio for img2img", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}),
    "token_merging_ratio_hr": OptionInfo(0.0, "Token merging ratio for hires", gr.Slider, {"minimum": 0.0, "maximum": 0.9, "step": 0.1}),

    "freeu_sep": OptionInfo("<h2>FreeU</h2>", "", gr.HTML),
    "freeu_enabled": OptionInfo(False, "FreeU"),
    "freeu_b1": OptionInfo(1.2, "1st stage backbone factor", gr.Slider, {"minimum": 1.0, "maximum": 2.0, "step": 0.01}),
    "freeu_b2": OptionInfo(1.4, "2nd stage backbone factor", gr.Slider, {"minimum": 1.0, "maximum": 2.0, "step": 0.01}),
    "freeu_s1": OptionInfo(0.9, "1st stage skip factor", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "freeu_s2": OptionInfo(0.2, "2nd stage skip factor", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),

    "hypertile_sep": OptionInfo("<h2>HyperTile</h2>", "", gr.HTML),
    "hypertile_unet_enabled": OptionInfo(False, "HyperTile UNet"),
    "hypertile_unet_tile": OptionInfo(256, "HyperTile UNet tile size", gr.Slider, {"minimum": 0, "maximum": 1024, "step": 8}),
    "hypertile_vae_enabled": OptionInfo(False, "HyperTile VAE", gr.Checkbox),
    "hypertile_vae_tile": OptionInfo(128, "HyperTile VAE tile size", gr.Slider, {"minimum": 0, "maximum": 1024, "step": 8}),

    "inference_other_sep": OptionInfo("<h2>Other</h2>", "", gr.HTML),
    "batch_frame_mode": OptionInfo(False, "Parallel process images in batch"),
    "inference_mode": OptionInfo("no-grad", "Torch inference mode", gr.Radio, {"choices": ["no-grad", "inference-mode", "none"]}),
    "sd_vae_sliced_encode": OptionInfo(False, "VAE sliced encode"),
}))

options_templates.update(options_section(('diffusers', "Diffusers Settings"), {
    "diffusers_pipeline": OptionInfo('Autodetect', 'Diffusers pipeline', gr.Dropdown, lambda: {"choices": list(shared_items.get_pipelines()) }),
    "diffusers_move_base": OptionInfo(False, "Move base model to CPU when using refiner"),
    "diffusers_move_unet": OptionInfo(False, "Move base model to CPU when using VAE"),
    "diffusers_move_refiner": OptionInfo(False, "Move refiner model to CPU when not in use"),
    "diffusers_extract_ema": OptionInfo(True, "Use model EMA weights when possible"),
    "diffusers_generator_device": OptionInfo("GPU", "Generator device", gr.Radio, {"choices": ["GPU", "CPU", "Unset"]}),
    "diffusers_model_cpu_offload": OptionInfo(False, "Model CPU offload (--medvram)"),
    "diffusers_seq_cpu_offload": OptionInfo(False, "Sequential CPU offload (--lowvram)"),
    "diffusers_vae_upcast": OptionInfo("default", "VAE upcasting", gr.Radio, {"choices": ['default', 'true', 'false']}),
    "diffusers_vae_slicing": OptionInfo(False, "VAE slicing"),
    "diffusers_vae_tiling": OptionInfo(False, "VAE tiling"),
    "diffusers_attention_slicing": OptionInfo(False, "Attention slicing"),
    "diffusers_model_load_variant": OptionInfo("default", "Preferred Model variant", gr.Radio, {"choices": ['default', 'fp32', 'fp16']}),
    "diffusers_vae_load_variant": OptionInfo("default", "Preferred VAE variant", gr.Radio, {"choices": ['default', 'fp32', 'fp16']}),
    "custom_diffusers_pipeline": OptionInfo('', 'Load custom Diffusers pipeline'),
    "diffusers_eval": OptionInfo(True, "Force model eval"),
    "diffusers_to_gpu": OptionInfo(False, "Load model directly to GPU"),
    "disable_accelerate": OptionInfo(False, "Disable accelerate"),
    "diffusers_force_zeros": OptionInfo(False, "Force zeros for prompts when empty", gr.Checkbox, {"visible": False}),
    "diffusers_aesthetics_score": OptionInfo(False, "Require aesthetics score"),
    "diffusers_pooled": OptionInfo("default", "Diffusers SDXL pooled embeds", gr.Radio, {"choices": ['default', 'weighted']}),
    "huggingface_token": OptionInfo('', 'HuggingFace token'),

    "onnx_sep": OptionInfo("<h2>ONNX Runtime</h2>", "", gr.HTML),
    "onnx_execution_provider": OptionInfo(get_default_execution_provider().value, 'Execution Provider', gr.Dropdown, lambda: {"choices": available_execution_providers }),
    "onnx_show_menu": OptionInfo(False, 'ONNX show onnx-specific menu (restart required)'),
    "onnx_cache_converted": OptionInfo(True, 'ONNX cache converted models'),
    "onnx_unload_base": OptionInfo(False, 'ONNX unload base model when processing refiner'),
}))

options_templates.update(options_section(('system-paths', "System Paths"), {
    "models_paths_sep_options": OptionInfo("<h2>Models paths</h2>", "", gr.HTML),
    "models_dir": OptionInfo('models', "Base path where all models are stored", folder=True),
    "ckpt_dir": OptionInfo(os.path.join(paths.models_path, 'Stable-diffusion'), "Folder with stable diffusion models", folder=True),
    "diffusers_dir": OptionInfo(os.path.join(paths.models_path, 'Diffusers'), "Folder with Huggingface models", folder=True),
    "hfcache_dir": OptionInfo(os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub'), "Folder for Huggingface cache", folder=True),
    "vae_dir": OptionInfo(os.path.join(paths.models_path, 'VAE'), "Folder with VAE files", folder=True),
    "sd_lora": OptionInfo("", "Add LoRA to prompt", gr.Textbox, {"visible": False}),
    "lora_dir": OptionInfo(os.path.join(paths.models_path, 'Lora'), "Folder with LoRA network(s)", folder=True),
    "lyco_dir": OptionInfo(os.path.join(paths.models_path, 'LyCORIS'), "Folder with LyCORIS network(s)", gr.Text, {"visible": False}),
    "styles_dir": OptionInfo(os.path.join(paths.data_path, 'styles.csv'), "File or Folder with user-defined styles", folder=True),
    "embeddings_dir": OptionInfo(os.path.join(paths.models_path, 'embeddings'), "Folder with textual inversion embeddings", folder=True),
    "hypernetwork_dir": OptionInfo(os.path.join(paths.models_path, 'hypernetworks'), "Folder with Hypernetwork models", folder=True),
    "control_dir": OptionInfo(os.path.join(paths.models_path, 'control'), "Folder with Control models", folder=True),
    "codeformer_models_path": OptionInfo(os.path.join(paths.models_path, 'Codeformer'), "Folder with codeformer models", folder=True),
    "gfpgan_models_path": OptionInfo(os.path.join(paths.models_path, 'GFPGAN'), "Folder with GFPGAN models", folder=True),
    "esrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'ESRGAN'), "Folder with ESRGAN models", folder=True),
    "bsrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'BSRGAN'), "Folder with BSRGAN models", folder=True),
    "realesrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'RealESRGAN'), "Folder with RealESRGAN models", folder=True),
    "scunet_models_path": OptionInfo(os.path.join(paths.models_path, 'SCUNet'), "Folder with SCUNet models", folder=True),
    "swinir_models_path": OptionInfo(os.path.join(paths.models_path, 'SwinIR'), "Folder with SwinIR models", folder=True),
    "ldsr_models_path": OptionInfo(os.path.join(paths.models_path, 'LDSR'), "Folder with LDSR models", folder=True),
    "clip_models_path": OptionInfo(os.path.join(paths.models_path, 'CLIP'), "Folder with CLIP models", folder=True),
    "onnx_cached_models_path": OptionInfo(os.path.join(paths.models_path, 'ONNX', 'cache'), "Folder with ONNX cached models", folder=True),

    "other_paths_sep_options": OptionInfo("<h2>Other paths</h2>", "", gr.HTML),
    "openvino_cache_path": OptionInfo('cache', "Directory for OpenVINO cache", folder=True),
    "temp_dir": OptionInfo("", "Directory for temporary images; leave empty for default", folder=True),
    "clean_temp_dir_at_start": OptionInfo(True, "Cleanup non-default temporary directory when starting webui"),
    "onnx_temp_dir": OptionInfo(os.path.join(paths.models_path, 'ONNX', 'temp'), "Directory for ONNX conversion and Olive optimization process", folder=True),
}))

options_templates.update(options_section(('saving-images', "Image Options"), {
    "keep_incomplete": OptionInfo(True, "Keep incomplete images"),
    "samples_save": OptionInfo(True, "Save all generated images"),
    "samples_format": OptionInfo('jpg', 'File format', gr.Dropdown, {"choices": ["jpg", "png", "webp", "tiff", "jp2"]}),
    "jpeg_quality": OptionInfo(90, "Image quality", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "img_max_size_mp": OptionInfo(250, "Maximum image size (MP)", gr.Slider, {"minimum": 100, "maximum": 2000, "step": 1}),
    "webp_lossless": OptionInfo(False, "WebP lossless compression"),
    "save_selected_only": OptionInfo(True, "Save only saves selected image"),
    "samples_save_zip": OptionInfo(True, "Create ZIP archive"),

    "image_sep_metadata": OptionInfo("<h2>Metadata/Logging</h2>", "", gr.HTML),
    "image_metadata": OptionInfo(True, "Include metadata"),
    "save_txt": OptionInfo(False, "Create info file per image"),
    "save_log_fn": OptionInfo("", "Update JSON log file per image", component_args=hide_dirs),
    "image_watermark_enabled": OptionInfo(False, "Include watermark"),
    "image_watermark": OptionInfo('', "Watermark string"),
    "image_sep_grid": OptionInfo("<h2>Grid Options</h2>", "", gr.HTML),
    "grid_save": OptionInfo(True, "Save all generated image grids"),
    "grid_format": OptionInfo('jpg', 'File format', gr.Dropdown, {"choices": ["jpg", "png", "webp", "tiff", "jp2"]}),
    "n_rows": OptionInfo(-1, "Row count", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    "grid_background": OptionInfo("#000000", "Background color", gr.ColorPicker, {}),
    "font": OptionInfo("", "Font file"),
    "font_color": OptionInfo("#FFFFFF", "Font color", gr.ColorPicker, {}),

    "save_sep_options": OptionInfo("<h2>Intermediate Image Saving</h2>", "", gr.HTML),
    "save_init_img": OptionInfo(False, "Save init images"),
    "save_images_before_highres_fix": OptionInfo(False, "Save image before hires"),
    "save_images_before_refiner": OptionInfo(False, "Save image before refiner"),
    "save_images_before_face_restoration": OptionInfo(False, "Save image before face restoration"),
    "save_images_before_color_correction": OptionInfo(False, "Save image before color correction"),
    "save_mask": OptionInfo(False, "Save inpainting mask"),
    "save_mask_composite": OptionInfo(False, "Save inpainting masked composite"),
}))

options_templates.update(options_section(('saving-paths', "Image Naming & Paths"), {
    "saving_sep_images": OptionInfo("<h2>Save options</h2>", "", gr.HTML),
    "save_images_add_number": OptionInfo(True, "Numbered filenames", component_args=hide_dirs),
    "use_original_name_batch": OptionInfo(True, "Batch uses original name"),
    "use_upscaler_name_as_suffix": OptionInfo(True, "Use upscaler as suffix", gr.Checkbox, {"visible": False}),
    "save_to_dirs": OptionInfo(False, "Save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs),
    "samples_filename_pattern": OptionInfo("[seq]-[model_name]-[prompt_words]", "Images filename pattern", component_args=hide_dirs),
    "directories_max_prompt_words": OptionInfo(8, "Max words per pattern", gr.Slider, {"minimum": 1, "maximum": 99, "step": 1, **hide_dirs}),
    "use_save_to_dirs_for_ui": OptionInfo(False, "Save images to a subdirectory when using Save button", gr.Checkbox, {"visible": False}),

    "outdir_sep_dirs": OptionInfo("<h2>Folders</h2>", "", gr.HTML),
    "outdir_samples": OptionInfo("", "Images folder", component_args=hide_dirs, folder=True),
    "outdir_txt2img_samples": OptionInfo("outputs/text", 'Folder for text generate', component_args=hide_dirs, folder=True),
    "outdir_img2img_samples": OptionInfo("outputs/image", 'Folder for image generate', component_args=hide_dirs, folder=True),
    "outdir_control_samples": OptionInfo("outputs/control", 'Folder for control generate', component_args=hide_dirs, folder=True),
    "outdir_extras_samples": OptionInfo("outputs/extras", 'Folder for processed images', component_args=hide_dirs, folder=True),
    "outdir_save": OptionInfo("outputs/save", "Folder for manually saved images", component_args=hide_dirs, folder=True),
    "outdir_video": OptionInfo("outputs/video", "Folder for videos", component_args=hide_dirs, folder=True),
    "outdir_init_images": OptionInfo("outputs/init-images", "Folder for init images", component_args=hide_dirs, folder=True),

    "outdir_sep_grids": OptionInfo("<h2>Grids</h2>", "", gr.HTML),
    "grid_extended_filename": OptionInfo(True, "Add extended info to filename when saving grid", gr.Checkbox, {"visible": False}),
    "grid_save_to_dirs": OptionInfo(False, "Save grids to a subdirectory", gr.Checkbox, {"visible": False}),
    "outdir_grids": OptionInfo("", "Grids folder", component_args=hide_dirs, folder=True),
    "outdir_txt2img_grids": OptionInfo("outputs/grids", 'Folder for txt2img grids', component_args=hide_dirs, folder=True),
    "outdir_img2img_grids": OptionInfo("outputs/grids", 'Folder for img2img grids', component_args=hide_dirs, folder=True),
    "outdir_control_grids": OptionInfo("outputs/grids", 'Folder for control grids', component_args=hide_dirs, folder=True),
}))

options_templates.update(options_section(('ui', "User Interface Options"), {
    "motd": OptionInfo(True, "Show MOTD"),
    "gradio_theme": OptionInfo("black-teal", "UI theme", gr.Dropdown, lambda: {"choices": theme.list_themes()}, refresh=theme.refresh_themes),
    "theme_style": OptionInfo("Auto", "Theme mode", gr.Radio, {"choices": ["Auto", "Dark", "Light"]}),
    "font_size": OptionInfo(14, "Font size", gr.Slider, {"minimum": 8, "maximum": 32, "step": 1, "visible": True}),
    "tooltips": OptionInfo("UI Tooltips", "UI tooltips", gr.Radio, {"choices": ["None", "Browser default", "UI tooltips"], "visible": False}),
    "compact_view": OptionInfo(False, "Compact view"),
    "return_grid": OptionInfo(True, "Show grid in results"),
    "return_mask": OptionInfo(False, "Inpainting include greyscale mask in results"),
    "return_mask_composite": OptionInfo(False, "Inpainting include masked composite in results"),
    "disable_weights_auto_swap": OptionInfo(True, "Do not change selected model when reading generation parameters"),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
    "keyedit_precision_attention": OptionInfo(0.1, "Ctrl+up/down precision when editing (attention:1.1)", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001, "visible": False}),
    "keyedit_precision_extra": OptionInfo(0.05, "Ctrl+up/down precision when editing <extra networks:0.9>", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001, "visible": False}),
    "keyedit_delimiters": OptionInfo(".,\/!?%^*;:{}=`~()", "Ctrl+up/down word delimiters", gr.Textbox, { "visible": False }), # pylint: disable=anomalous-backslash-in-string
    "quicksettings_list": OptionInfo(["sd_model_checkpoint"] if backend == Backend.ORIGINAL else ["sd_model_checkpoint", "sd_model_refiner"], "Quicksettings list", gr.Dropdown, lambda: {"multiselect":True, "choices": list(opts.data_labels.keys())}),
    "ui_scripts_reorder": OptionInfo("", "UI scripts order", gr.Textbox, { "visible": False }),
}))

options_templates.update(options_section(('live-preview', "Live Previews"), {
    "show_progressbar": OptionInfo(True, "Show progressbar", gr.Checkbox, {"visible": False}),
    "live_previews_enable": OptionInfo(True, "Show live previews", gr.Checkbox, {"visible": False}),
    "show_progress_grid": OptionInfo(True, "Show previews as a grid", gr.Checkbox, {"visible": False}),
    "notification_audio_enable": OptionInfo(False, "Play a notification upon completion"),
    "notification_audio_path": OptionInfo("html/notification.mp3","Path to notification sound", component_args=hide_dirs, folder=True),
    "show_progress_every_n_steps": OptionInfo(1, "Live preview display period", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
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
    "schedulers_prediction_type": OptionInfo("default", "Override model prediction type", gr.Radio, {"choices": ['default', 'epsilon', 'sample', 'v_prediction']}),

    # managed from ui.py for backend diffusers
    "schedulers_sep_diffusers": OptionInfo("<h2>Diffusers specific config</h2>", "", gr.HTML),
    "schedulers_dpm_solver": OptionInfo("sde-dpmsolver++", "DPM solver algorithm", gr.Radio, {"choices": ['dpmsolver++', 'sde-dpmsolver++']}),
    "schedulers_beta_schedule": OptionInfo("default", "Beta schedule", gr.Radio, {"choices": ['default', 'linear', 'scaled_linear', 'squaredcos_cap_v2']}),
    'schedulers_beta_start': OptionInfo(0, "Beta start", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.00001}),
    'schedulers_beta_end': OptionInfo(0, "Beta end", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.00001}),
    'schedulers_timesteps_range': OptionInfo(1000, "Timesteps range", gr.Slider, {"minimum": 250, "maximum": 4000, "step": 1}),
    "schedulers_rescale_betas": OptionInfo(False, "Rescale betas with zero terminal SNR", gr.Checkbox),

    # managed from ui.py for backend original k-diffusion
    "schedulers_sep_kdiffusers": OptionInfo("<h2>K-Diffusion specific config</h2>", "", gr.HTML),
    "always_batch_cond_uncond": OptionInfo(False, "Disable conditional batching"),
    "enable_quantization": OptionInfo(True, "Use quantization"),
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
    # TODO pad_cond_uncond implementation missing for original backend
    "pad_cond_uncond": OptionInfo(True, "Pad prompt and negative prompt to be same length", gr.Checkbox, {"visible": False}),
    # TODO batch_cond-uncond implementation missing for original backend
    "batch_cond_uncond": OptionInfo(True, "Do conditional and unconditional denoising in one batch", gr.Checkbox, {"visible": False}),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Additional postprocessing operations", gr.Dropdown, lambda: {"multiselect":True, "choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", gr.Dropdown, lambda: {"multiselect":True, "choices": [x.name for x in shared_items.postprocessing_scripts()]}),

    "postprocessing_sep_img2img": OptionInfo("<h2>Img2Img & Inpainting</h2>", "", gr.HTML),
    "img2img_color_correction": OptionInfo(False, "Apply color correction"),
    "img2img_fix_steps": OptionInfo(False, "For image processing do exact number of steps as specified", gr.Checkbox, { "visible": False }),
    "img2img_background_color": OptionInfo("#ffffff", "Image transparent color fill", gr.ColorPicker, {}),
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for image processing", gr.Slider, {"minimum": 0.1, "maximum": 1.5, "step": 0.01}),
    "img2img_extra_noise": OptionInfo(0.0, "Extra noise multiplier for img2img", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 8, "step": 1, "visible": False}),

    "postprocessing_sep_face_restoration": OptionInfo("<h2>Face Restoration</h2>", "", gr.HTML),
    "face_restoration_model": OptionInfo("CodeFormer", "Face restoration model", gr.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.2, "CodeFormer weight parameter", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model to CPU when complete"),

    "postprocessing_sep_upscalers": OptionInfo("<h2>Upscaling</h2>", "", gr.HTML),
    "upscaler_unload": OptionInfo(False, "Unload upscaler after processing"),
    "upscaler_for_img2img": OptionInfo("None", "Default upscaler for image resize operations", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers], "visible": False}, refresh=refresh_upscalers),
    "upscaler_tile_size": OptionInfo(192, "Upscaler tile size", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "upscaler_tile_overlap": OptionInfo(8, "Upscaler tile overlap", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}),
}))

options_templates.update(options_section(('control', "Control Options"), {
    "control_max_units": OptionInfo(3, "Maximum number of units", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
    "control_move_processor": OptionInfo(False, "Processor move to CPU after use"),
    "control_unload_processor": OptionInfo(False, "Processor unload after use"),
}))

options_templates.update(options_section(('training', "Training"), {
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training"),
    "pin_memory": OptionInfo(True, "Pin training dataset to memory"),
    "save_optimizer_state": OptionInfo(False, "Save resumable optimizer state when training"),
    "save_training_settings_to_txt": OptionInfo(True, "Save training settings to a text file"),
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
    "extra_networks": OptionInfo(["All"], "Extra networks", gr.Dropdown, lambda: {"multiselect":True, "choices": ['All'] + [en.title for en in extra_networks]}),
    "extra_networks_view": OptionInfo("gallery", "UI view", gr.Radio, {"choices": ["gallery", "list"]}),
    "extra_networks_card_cover": OptionInfo("sidebar", "UI position", gr.Radio, {"choices": ["cover", "inline", "sidebar"]}),
    "extra_networks_height": OptionInfo(53, "UI height (%)", gr.Slider, {"minimum": 10, "maximum": 100, "step": 1}),
    "extra_networks_sidebar_width": OptionInfo(35, "UI sidebar width (%)", gr.Slider, {"minimum": 10, "maximum": 80, "step": 1}),
    "extra_networks_card_size": OptionInfo(160, "UI card size (px)", gr.Slider, {"minimum": 20, "maximum": 2000, "step": 1}),
    "extra_networks_card_square": OptionInfo(True, "UI disable variable aspect ratio"),
    "extra_networks_card_fit": OptionInfo("cover", "UI image contain method", gr.Radio, {"choices": ["contain", "cover", "fill"], "visible": False}),
    "extra_networks_sep2": OptionInfo("<h2>Extra networks general</h2>", "", gr.HTML),
    "extra_network_skip_indexing": OptionInfo(False, "Build info on first access", gr.Checkbox),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Default multiplier for extra networks", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "diffusers_convert_embed": OptionInfo(False, "Auto-convert SD 1.5 embeddings to SDXL ", gr.Checkbox, {"visible": backend==Backend.DIFFUSERS}),
    "extra_networks_sep3": OptionInfo("<h2>Extra networks settings</h2>", "", gr.HTML),
    "extra_networks_styles": OptionInfo(True, "Show built-in styles"),
    "lora_preferred_name": OptionInfo("filename", "LoRA preffered name", gr.Radio, {"choices": ["filename", "alias"]}),
    "lora_add_hashes_to_infotext": OptionInfo(True, "LoRA add hash info"),
    "lora_force_diffusers": OptionInfo(False if not cmd_opts.use_openvino else True, "LoRA use alternative loading method"),
    "lora_fuse_diffusers": OptionInfo(False if not cmd_opts.use_openvino else True, "LoRA use merge when using alternative method"),
    "lora_in_memory_limit": OptionInfo(1, "LoRA memory cache", gr.Slider, {"minimum": 0, "maximum": 24, "step": 1}),
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
                errors.display(e, 'Error in onchange callback')
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

            if os.environ.get('SD_CONFIG_DEBUG', None) is not None:
                log.debug('Config: user settings')
                for k, v in self.data.items():
                    log.trace(f'  Config: item={k} value={v} default={self.data_labels[k].default if k in self.data_labels else None}')
                log.debug('Config: default settings')
                for k in self.data_labels.keys():
                    log.trace(f'  Config: item={k} default={self.data_labels[k].default}')

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
        self.data = readfile(filename, lock=True)
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

profiler = None
opts = Options()
config_filename = cmd_opts.config
opts.load(config_filename)
cmd_opts = cmd_args.compatibility_args(opts, cmd_opts)
if cmd_opts.use_xformers:
    opts.data['cross_attention_optimization'] = 'xFormers'
opts.data['uni_pc_lower_order_final'] = opts.schedulers_use_loworder # compatibility
opts.data['uni_pc_order'] = opts.schedulers_solver_order # compatibility
log.info(f'Engine: backend={backend} compute={devices.backend} device={devices.get_optimal_device_name()} attention="{opts.cross_attention_optimization}" mode={devices.inference_context.__name__}')
log.info(f'Device: {print_dict(devices.get_gpu_info())}')

prompt_styles = modules.styles.StyleDatabase(opts)
cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or (cmd_opts.server_name or False)) and not cmd_opts.insecure
devices.device, devices.device_interrogate, devices.device_gfpgan, devices.device_esrgan, devices.device_codeformer = (devices.cpu if any(y in cmd_opts.use_cpu for y in [x, 'all']) else devices.get_optimal_device() for x in ['sd', 'interrogate', 'gfpgan', 'esrgan', 'codeformer'])
device = devices.device
batch_cond_uncond = opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram
mem_mon = modules.memmon.MemUsageMonitor("MemMon", devices.device)
max_workers = 4
if devices.backend == "directml":
    directml_do_hijack()
initialize_onnx()


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


def listdir(path):
    if not os.path.exists(path):
        return []
    mtime = os.path.getmtime(path)
    if path in dir_timestamps and mtime == dir_timestamps[path]:
        return dir_cache[path]
    else:
        dir_cache[path] = [os.path.join(path, f) for f in os.listdir(path)]
        dir_timestamps[path] = mtime
        return dir_cache[path]


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


sd_model: diffusers.DiffusionPipeline = None # dummy and overwritten by class
sd_refiner: diffusers.DiffusionPipeline = None # dummy and overwritten by class
sd_model_type: str = '' # dummy and overwritten by class
sd_refiner_type: str = '' # dummy and overwritten by class
compiled_model_state = None

from modules.modeldata import Shared # pylint: disable=ungrouped-imports
sys.modules[__name__].__class__ = Shared
