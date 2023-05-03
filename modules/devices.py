import sys
import contextlib
import torch
from modules import cmd_args, shared
try:
    import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
except:
    pass

if sys.platform == "darwin":
    from modules import mac_specific # pylint: disable=ungrouped-imports


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps

def extract_device_id(args, name): # pylint: disable=redefined-outer-name
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]
    return None


def get_cuda_device_string():
    if shared.cmd_opts.use_ipex:
        return "xpu"
    else:
        if shared.cmd_opts.device_id is not None:
            return f"cuda:{shared.cmd_opts.device_id}"
        return "cuda"


def get_dml_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"privateuseone:{shared.cmd_opts.device_id}"
    return "privateuseone:0"


def get_optimal_device_name():
    if shared.cmd_opts.use_ipex:
        return "xpu"
    elif torch.cuda.is_available():
        return get_cuda_device_string()
    if has_mps():
        return "mps"
    try:
        import torch_directml # pylint: disable=import-error
        if torch_directml.is_available():
            return get_dml_device_string()
        else:
            return "cpu"
    except:
        return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    if task in shared.cmd_opts.use_cpu:
        return cpu
    return get_optimal_device()


def torch_gc():
    if shared.cmd_opts.use_ipex:
        with torch.xpu.device("xpu"):
            torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def set_cuda_params():
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = shared.opts.cuda_allow_tf32
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = shared.opts.cuda_allow_tf16_reduced
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = shared.opts.cuda_allow_tf16_reduced
        except:
            pass
        if torch.backends.cudnn.is_available():
            try:
                torch.backends.cudnn.benchmark = shared.opts.cudnn_benchmark
                torch.backends.cudnn.benchmark_limit = 0
                torch.backends.cudnn.allow_tf32 = shared.opts.cuda_allow_tf32
            except:
                pass
    global dtype, dtype_vae, dtype_unet, unet_needs_upcast # pylint: disable=global-statement
    # set dtype
    if shared.opts.cuda_dtype == 'FP16':
        dtype = torch.float16
        dtype_vae = torch.float16
        dtype_unet = torch.float16
    if shared.opts.cuda_dtype == 'BP16':
        dtype = torch.bfloat16
        dtype_vae = torch.bfloat16
        dtype_unet = torch.bfloat16
    if shared.opts.cuda_dtype == 'FP32' or shared.opts.no_half:
        dtype = torch.float32
        dtype_vae = torch.float32
        dtype_unet = torch.float32
    if shared.opts.no_half_vae: # set dtype again as no-half-vae options take priority
        dtype_vae = torch.float32
    unet_needs_upcast = shared.opts.upcast_sampling

args = cmd_args.parser.parse_args()
if args.use_ipex:
    cpu = torch.device("xpu") #Use XPU instead of CPU. %20 Perf improvement on weak CPUs.
    print("Using XPU instead of CPU.")
else:
    cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16
dtype_unet = torch.float16
unet_needs_upcast = False


def cond_cast_unet(tensor):
    return tensor.to(dtype_unet) if unet_needs_upcast else tensor


def cond_cast_float(tensor):
    return tensor.float() if unet_needs_upcast else tensor


def randn(seed, shape):
    torch.manual_seed(seed)
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def autocast(disable=False):
    if disable:
        return contextlib.nullcontext()
    if dtype == torch.float32 or shared.cmd_opts.precision == "Full":
        return contextlib.nullcontext()
    if shared.cmd_opts.use_ipex:
        return torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)
    else:
        return torch.autocast("cuda")


def without_autocast(disable=False):
    if shared.cmd_opts.use_ipex:
        return torch.autocast("xpu", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()
    else:
        return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if shared.opts.disable_nan_check:
        return
    if not torch.all(torch.isnan(x)).item():
        return
    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."
        if not shared.cmd_opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."
    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."
        if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."
    message += " Use --disable-nan-check commandline argument to disable this check."
    raise NansException(message)
