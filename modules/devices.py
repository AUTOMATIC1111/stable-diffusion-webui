import gc
import sys
import contextlib
import torch
from modules import cmd_args, shared, memstats

if sys.platform == "darwin":
    from modules import mac_specific # pylint: disable=ungrouped-imports

cuda_ok = torch.cuda.is_available()

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


def get_optimal_device_name():
    if shared.cmd_opts.use_ipex:
        return "xpu"
    elif cuda_ok and not shared.cmd_opts.use_directml:
        return get_cuda_device_string()
    if has_mps():
        return "mps"
    if shared.cmd_opts.use_directml:
        import torch_directml # pylint: disable=import-error
        if torch_directml.is_available():
            torch.cuda.is_available = lambda: False
            if shared.cmd_opts.device_id is not None:
                return f"privateuseone:{shared.cmd_opts.device_id}"
            return torch_directml.device()
        else:
            return "cpu"
    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    if task in shared.cmd_opts.use_cpu:
        return cpu
    return get_optimal_device()


def torch_gc(force=False):
    if shared.opts.disable_gc and not force:
        return
    collected = gc.collect()
    if shared.cmd_opts.use_ipex:
        try:
            with torch.xpu.device("xpu"):
                torch.xpu.empty_cache()
        except:
            pass
    elif cuda_ok:
        try:
            with torch.cuda.device(get_cuda_device_string()):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except:
            pass
    shared.log.debug(f'gc: collected={collected} device={torch.device(get_optimal_device_name())} {memstats.memory_stats()}')


def test_fp16():
    if shared.cmd_opts.experimental:
        return True
    try:
        x = torch.tensor([[1.5,.0,.0,.0]]).to(device).half()
        layerNorm = torch.nn.LayerNorm(4, eps=0.00001, elementwise_affine=True, dtype=torch.float16, device=device)
        _y = layerNorm(x)
        shared.log.debug('Torch FP16 test passed')
        return True
    except:
        shared.log.warning('Torch FP16 test failed: Forcing FP32 operations')
        shared.opts.cuda_dtype = 'FP32'
        shared.opts.no_half = True
        shared.opts.no_half_vae = True
        return False


def set_cuda_params():
    shared.log.debug('Verifying Torch settings')
    if cuda_ok:
        try:
            torch.backends.cuda.matmul.allow_tf32 = shared.opts.cuda_allow_tf32
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = shared.opts.cuda_allow_tf16_reduced
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = shared.opts.cuda_allow_tf16_reduced
        except:
            pass
        if torch.backends.cudnn.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                if shared.opts.cudnn_benchmark:
                    torch.backends.cudnn.benchmark_limit = 0
                torch.backends.cudnn.allow_tf32 = shared.opts.cuda_allow_tf32
            except:
                pass
    global dtype, dtype_vae, dtype_unet, unet_needs_upcast # pylint: disable=global-statement
    ok = test_fp16()
    if shared.cmd_opts.use_directml and not shared.cmd_opts.experimental: # TODO DirectML does not have full autocast capabilities
        shared.opts.no_half = True
        shared.opts.no_half_vae = True
    if ok and shared.opts.cuda_dtype == 'FP32':
        shared.log.info('CUDA FP16 test passed but desired mode is set to FP32')
    if shared.opts.cuda_dtype == 'FP16' and ok:
        dtype = torch.float16
        dtype_vae = torch.float16
        dtype_unet = torch.float16
    if shared.opts.cuda_dtype == 'BP16' and ok:
        dtype = torch.bfloat16
        dtype_vae = torch.bfloat16
        dtype_unet = torch.bfloat16
    if shared.opts.cuda_dtype == 'FP32' or shared.opts.no_half or not ok:
        dtype = torch.float32
        dtype_vae = torch.float32
        dtype_unet = torch.float32
    if shared.opts.no_half_vae: # set dtype again as no-half-vae options take priority
        dtype_vae = torch.float32
    unet_needs_upcast = shared.opts.upcast_sampling
    shared.log.debug(f'Desired Torch parameters: dtype={shared.opts.cuda_dtype} no-half={shared.opts.no_half} no-half-vae={shared.opts.no_half_vae} upscast={shared.opts.upcast_sampling}')
    shared.log.info(f'Setting Torch parameters: dtype={dtype} vae={dtype_vae} unet={dtype_unet}')
    shared.log.debug(f'Torch default device: {torch.device(get_optimal_device_name())}')


args = cmd_args.parser.parse_args()
if args.use_ipex:
    cpu = torch.device("xpu") #Use XPU instead of CPU. %20 Perf improvement on weak CPUs.
else:
    cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16
dtype_unet = torch.float16
unet_needs_upcast = False
if args.use_ipex:
    backend = 'ipex'
elif args.use_directml:
    backend = 'directml'
elif torch.cuda.is_available() and torch.version.cuda:
    backend = 'cuda'
elif torch.cuda.is_available() and torch.version.hip:
    backend = 'rocm'
elif sys.platform == 'darwin':
    backend = 'mps'
else:
    backend = 'cpu'



def cond_cast_unet(tensor):
    return tensor.to(dtype_unet) if unet_needs_upcast else tensor


def cond_cast_float(tensor):
    return tensor.float() if unet_needs_upcast else tensor


def randn(seed, shape):
    torch.manual_seed(seed)
    if shared.cmd_opts.use_ipex:
        torch.xpu.manual_seed_all(seed)
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
    if shared.cmd_opts.use_directml:
        return torch.dml.amp.autocast(dtype)
    if shared.cmd_opts.use_ipex:
        return torch.xpu.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)
    if cuda_ok:
        return torch.autocast("cuda")
    else:
        return torch.autocast("cpu")


def without_autocast(disable=False):
    if disable:
        return contextlib.nullcontext()
    if shared.cmd_opts.use_directml:
        return torch.dml.amp.autocast(enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext()
    if shared.cmd_opts.use_ipex:
        return torch.xpu.amp.autocast(enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext()
    if cuda_ok:
        return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext()
    else:
        return torch.autocast("cpu", enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if shared.opts.disable_nan_check:
        return
    if not torch.all(torch.isnan(x)).item():
        return
    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."
        if not shared.opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."
    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."
        if not shared.opts.no_half and not shared.opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."
    message += " Use --disable-nan-check commandline argument to disable this check."
    raise NansException(message)
