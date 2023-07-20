import gc
import sys
import contextlib
import torch
from modules import cmd_args, shared, memstats
from modules.dml import directml_init

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
        return torch.dml.get_default_device_string()
    else:
        if shared.cmd_opts.device_id is not None:
            return f"cuda:{shared.cmd_opts.device_id}"
        return "cuda"


def get_optimal_device_name():
    if cuda_ok or backend == 'ipex' or backend == 'directml':
        return get_cuda_device_string()
    if has_mps():
        return "mps"
    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    if task in shared.cmd_opts.use_cpu:
        return cpu
    return get_optimal_device()


def torch_gc(force=False):
    mem = memstats.memory_stats()
    gpu = mem.get('gpu', {})
    oom = gpu.get('oom', 0)
    used = round(100 * gpu.get('used', 0) / gpu.get('total', 1))
    global previous_oom # pylint: disable=global-statement
    if oom > previous_oom:
        previous_oom = oom
        shared.log.warning(f'GPU out-of-memory error: {mem}')
    if used > 95:
        shared.log.warning(f'GPU high memory utilization: {used}% {mem}')
        force = True

    if shared.opts.disable_gc and not force:
        return
    collected = gc.collect()
    if cuda_ok or backend == 'ipex':
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
            torch.backends.cuda.matmul.allow_tf32 = shared.opts.cuda_allow_tf32
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = shared.opts.cuda_allow_tf16_reduced
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = shared.opts.cuda_allow_tf16_reduced
        except Exception:
            pass
        if torch.backends.cudnn.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                if shared.opts.cudnn_benchmark:
                    shared.log.debug('Torch enable cuDNN benchmark')
                    torch.backends.cudnn.benchmark_limit = 0
                torch.backends.cudnn.allow_tf32 = shared.opts.cuda_allow_tf32
            except Exception:
                pass
    global dtype, dtype_vae, dtype_unet, unet_needs_upcast # pylint: disable=global-statement
    if shared.cmd_opts.use_directml and not shared.cmd_opts.experimental: # TODO DirectML does not have full autocast capabilities
        shared.opts.no_half = True
        shared.opts.no_half_vae = True
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

if backend == 'ipex':
    import os
    def ipex_no_cuda(orig_func, *args, **kwargs):
        torch.cuda.is_available = lambda: False
        orig_func(*args, **kwargs)
        torch.cuda.is_available = torch.xpu.is_available

    #Fix functions with ipex
    torch.cuda.is_available = torch.xpu.is_available
    torch.cuda.device = torch.xpu.device
    torch.cuda.device_count = torch.xpu.device_count
    torch.cuda.current_device = torch.xpu.current_device
    torch.cuda.get_device_name = torch.xpu.get_device_name
    torch.cuda.get_device_properties = torch.xpu.get_device_properties
    torch._utils._get_available_device_type = lambda: "xpu"
    torch.cuda.set_device = torch.xpu.set_device

    torch.cuda.empty_cache = torch.xpu.empty_cache if "WSL2" not in os.popen("uname -a").read() else lambda: None
    torch.cuda.ipc_collect = lambda: None
    torch.cuda.memory_stats = torch.xpu.memory_stats
    torch.cuda.mem_get_info = lambda device=None: [(torch.xpu.get_device_properties(device).total_memory - torch.xpu.memory_allocated(device)), torch.xpu.get_device_properties(device).total_memory]
    torch.cuda.memory_allocated = torch.xpu.memory_allocated
    torch.cuda.max_memory_allocated = torch.xpu.max_memory_allocated
    torch.cuda.reset_peak_memory_stats = torch.xpu.reset_peak_memory_stats
    torch.cuda.utilization = lambda: 0

    torch.cuda.get_rng_state_all = torch.xpu.get_rng_state_all
    torch.cuda.set_rng_state_all = torch.xpu.set_rng_state_all
    try:
        torch.cuda.amp.GradScaler = torch.xpu.amp.GradScaler
    except Exception:
        pass

    from modules.sd_hijack_utils import CondFunc
    #Broken functions when torch.cuda.is_available is True:
    CondFunc('torch.utils.data.dataloader._BaseDataLoaderIter.__init__',
        lambda orig_func, *args, **kwargs: ipex_no_cuda(orig_func, *args, **kwargs),
        lambda orig_func, *args, **kwargs: True)

    #Functions with dtype errors:
    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, *args, **kwargs: orig_func(args[0], args[1].to(args[0].weight.data.dtype)),
        lambda *args, **kwargs: args[2].dtype != args[1].weight.data.dtype)
    CondFunc('torch.nn.modules.Linear.forward',
        lambda orig_func, *args, **kwargs: orig_func(args[0], args[1].to(args[0].weight.data.dtype)),
        lambda *args, **kwargs: args[2].dtype != args[1].weight.data.dtype)
    #Diffusers bfloat16:
    CondFunc('torch.nn.modules.Conv2d._conv_forward',
        lambda orig_func, *args, **kwargs: orig_func(args[0], args[1].to(args[2].data.dtype), args[2], args[3]),
        lambda *args, **kwargs: args[2].dtype != args[3].data.dtype)

    #Functions that does not work with the XPU:
    #UniPC:
    CondFunc('torch.linalg.solve',
        lambda orig_func, *args, **kwargs: orig_func(args[0].to("cpu"), args[1].to("cpu")).to(get_cuda_device_string()),
        lambda *args, **kwargs: args[1].device != torch.device("cpu"))
    #SDE Samplers:
    CondFunc('torch.Generator',
        lambda orig_func, device: torch.xpu.Generator(device),
        lambda orig_func, device: device != torch.device("cpu") and device != "cpu")
    #Latent antialias:
    CondFunc('torch.nn.functional.interpolate',
        lambda orig_func, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False: orig_func(input.to("cpu"), size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias).to(get_cuda_device_string()),
        lambda orig_func, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False: antialias)
    #Diffusers Float64 (ARC GPUs doesn't support double or Float64):
    if not torch.xpu.has_fp64_dtype():
        CondFunc('torch.from_numpy',
            lambda orig_func, *args, **kwargs: orig_func(args[0].astype('float32')),
            lambda *args, **kwargs: args[1].dtype == float)
    #ControlNet:
    CondFunc('torch.batch_norm',
        lambda orig_func, *args, **kwargs: orig_func(args[0].to("cpu"),
        args[1].to("cpu") if args[1] is not None else args[1],
        args[2].to("cpu") if args[2] is not None else args[2],
        args[3].to("cpu") if args[3] is not None else args[3],
        args[4].to("cpu") if args[4] is not None else args[4],
        args[5], args[6], args[7], args[8]).to(get_cuda_device_string()),
        lambda *args, **kwargs: args[1].device != torch.device("cpu"))
    CondFunc('torch.instance_norm',
        lambda orig_func, *args, **kwargs: orig_func(args[0].to("cpu"),
        args[1].to("cpu") if args[1] is not None else args[1],
        args[2].to("cpu") if args[2] is not None else args[2],
        args[3].to("cpu") if args[3] is not None else args[3],
        args[4].to("cpu") if args[4] is not None else args[4],
        args[5], args[6], args[7], args[8]).to(get_cuda_device_string()),
        lambda *args, **kwargs: args[1].device != torch.device("cpu"))

if backend == "directml":
    directml_init()

cuda_ok = torch.cuda.is_available() and not backend == 'ipex'
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
    if backend == 'ipex':
        return torch.xpu.amp.autocast(enabled=True, dtype=dtype)
    if cuda_ok:
        return torch.autocast("cuda")
    else:
        return torch.autocast("cpu")


def without_autocast(disable=False):
    if disable:
        return contextlib.nullcontext()
    if shared.cmd_opts.use_directml:
        return torch.dml.amp.autocast(enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext()
    if backend == 'ipex':
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
