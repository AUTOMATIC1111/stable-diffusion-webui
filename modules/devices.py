import gc
import sys
import contextlib
import torch
from modules import cmd_args, shared, memstats

if sys.platform == "darwin":
    from modules import mac_specific # pylint: disable=ungrouped-imports

previous_oom = 0


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
    if backend == 'ipex':
        if shared.cmd_opts.device_id is not None:
            return f"xpu:{shared.cmd_opts.device_id}"
        return "xpu"
    elif backend == 'directml' and torch.dml.is_available():
        if shared.cmd_opts.device_id is not None:
            return f"privateuseone:{shared.cmd_opts.device_id}"
        return torch.dml.get_device_string(torch.dml.default_device().index)
    else:
        if shared.cmd_opts.device_id is not None:
            return f"cuda:{shared.cmd_opts.device_id}"
        return "cuda"


def get_optimal_device_name():
    if cuda_ok or backend == 'directml':
        return get_cuda_device_string()
    if has_mps():
        return "mps"
    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    if task in shared.cmd_opts.use_cpu:
        shared.log.debug(f'Forcing CPU for task: {task}')
        return cpu
    return get_optimal_device()


def torch_gc(force=False):
    mem = memstats.memory_stats()
    gpu = mem.get('gpu', {})
    oom = gpu.get('oom', 0)
    if backend == "directml":
        used = round(100 * torch.cuda.memory_allocated() / (1 << 30) / gpu.get('total', 1)) if gpu.get('total', 1) > 1 else 0
    else:
        used = round(100 * gpu.get('used', 0) / gpu.get('total', 1)) if gpu.get('total', 1) > 1 else 0
    global previous_oom # pylint: disable=global-statement
    if oom > previous_oom:
        previous_oom = oom
        shared.log.warning(f'GPU out-of-memory error: {mem}')
    if used > 95:
        shared.log.info(f'GPU high memory utilization: {used}% {mem}')
        force = True
    if not force:
        return
    collected = gc.collect()
    if cuda_ok:
        try:
            with torch.cuda.device(get_cuda_device_string()):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
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
    except Exception as e:
        shared.log.warning(f'Torch FP16 test failed: Forcing FP32 operations: {e}')
        shared.opts.cuda_dtype = 'FP32'
        shared.opts.no_half = True
        shared.opts.no_half_vae = True
        return False

def test_bf16():
    if shared.cmd_opts.experimental:
        return True
    try:
        import torch.nn.functional as F
        image = torch.randn(1, 4, 32, 32).to(device=device, dtype=torch.bfloat16)
        _out = F.interpolate(image, size=(64, 64), mode="nearest")
        return True
    except Exception:
        shared.log.warning('Torch BF16 test failed: Fallback to FP16 operations')
        return False


def set_cuda_params():
    shared.log.debug('Verifying Torch settings')
    if cuda_ok:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        except Exception:
            pass
        if torch.backends.cudnn.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                if shared.opts.cudnn_benchmark:
                    shared.log.debug('Torch enable cuDNN benchmark')
                    torch.backends.cudnn.benchmark_limit = 0
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
    global dtype, dtype_vae, dtype_unet, unet_needs_upcast # pylint: disable=global-statement
    if shared.opts.cuda_dtype == 'FP32':
        dtype = torch.float32
        dtype_vae = torch.float32
        dtype_unet = torch.float32
    if shared.opts.cuda_dtype == 'BF16' or dtype == torch.bfloat16:
        bf16_ok = test_bf16()
        dtype = torch.bfloat16 if bf16_ok else torch.float16
        dtype_vae = torch.bfloat16 if bf16_ok else torch.float16
        dtype_unet = torch.bfloat16 if bf16_ok else torch.float16
    if shared.opts.cuda_dtype == 'FP16' or dtype == torch.float16:
        fp16_ok = test_fp16()
        dtype = torch.float16 if fp16_ok else torch.float32
        dtype_vae = torch.float16 if fp16_ok else torch.float32
        dtype_unet = torch.float16 if fp16_ok else torch.float32
    else:
        pass
    if shared.opts.no_half:
        shared.log.info('Torch override dtype: no-half set')
        dtype = torch.float32
        dtype_vae = torch.float32
        dtype_unet = torch.float32
    if shared.opts.no_half_vae: # set dtype again as no-half-vae options take priority
        shared.log.info('Torch override VAE dtype: no-half set')
        dtype_vae = torch.float32
    unet_needs_upcast = shared.opts.upcast_sampling
    shared.log.debug(f'Desired Torch parameters: dtype={shared.opts.cuda_dtype} no-half={shared.opts.no_half} no-half-vae={shared.opts.no_half_vae} upscast={shared.opts.upcast_sampling}')
    shared.log.info(f'Setting Torch parameters: dtype={dtype} vae={dtype_vae} unet={dtype_unet}')
    shared.log.debug(f'Torch default device: {torch.device(get_optimal_device_name())}')


args = cmd_args.parser.parse_args()
if args.use_ipex or (hasattr(torch, 'xpu') and torch.xpu.is_available()):
    backend = 'ipex'
    from modules.intel.ipex import ipex_init
    ipex_init()
elif args.use_directml:
    backend = 'directml'
    from modules.dml import directml_init
    directml_init()
elif torch.cuda.is_available() and torch.version.cuda:
    backend = 'cuda'
elif torch.cuda.is_available() and torch.version.hip:
    backend = 'rocm'
elif sys.platform == 'darwin':
    backend = 'mps'
else:
    backend = 'cpu'

cuda_ok = torch.cuda.is_available()
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
    if backend == 'ipex':
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
    if cuda_ok:
        return torch.autocast("cuda")
    else:
        return torch.autocast("cpu")


def without_autocast(disable=False):
    if disable:
        return contextlib.nullcontext()
    if shared.cmd_opts.use_directml:
        return torch.dml.amp.autocast(enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext() # pylint: disable=unexpected-keyword-arg
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
