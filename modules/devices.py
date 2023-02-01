import sys, os, shlex
import contextlib
import torch
from modules import errors
from modules.sd_hijack_utils import CondFunc
from packaging import version


# has_mps is only available in nightly pytorch (for now) and macOS 12.3+.
# check `getattr` and try it for compatibility
def has_mps() -> bool:
    if not getattr(torch, 'has_mps', False):
        return False
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False


def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]

    return None


def get_cuda_device_string():
    from modules import shared

    if shared.cmd_opts.device_id is not None:
        return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    from modules import shared

    if task in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():

        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if any([torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())]):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True



errors.run(enable_tf32, "Enabling TF32")

cpu = torch.device("cpu")
device = device_interrogate = device_gfpgan = device_esrgan = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16
dtype_unet = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


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
    from modules import shared

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    from modules import shared

    if shared.cmd_opts.disable_nan_check:
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


# MPS workaround for https://github.com/pytorch/pytorch/issues/89784
def cumsum_fix(input, cumsum_func, *args, **kwargs):
    if input.device.type == 'mps':
        output_dtype = kwargs.get('dtype', input.dtype)
        if output_dtype == torch.int64:
            return cumsum_func(input.cpu(), *args, **kwargs).to(input.device)
        elif cumsum_needs_bool_fix and output_dtype == torch.bool or cumsum_needs_int_fix and (output_dtype == torch.int8 or output_dtype == torch.int16):
            return cumsum_func(input.to(torch.int32), *args, **kwargs).to(torch.int64)
    return cumsum_func(input, *args, **kwargs)


if has_mps():
    if version.parse(torch.__version__) < version.parse("1.13"):
        # PyTorch 1.13 doesn't need these fixes but unfortunately is slower and has regressions that prevent training from working

        # MPS workaround for https://github.com/pytorch/pytorch/issues/79383
        CondFunc('torch.Tensor.to', lambda orig_func, self, *args, **kwargs: orig_func(self.contiguous(), *args, **kwargs),
                                                          lambda _, self, *args, **kwargs: self.device.type != 'mps' and (args and isinstance(args[0], torch.device) and args[0].type == 'mps' or isinstance(kwargs.get('device'), torch.device) and kwargs['device'].type == 'mps'))
        # MPS workaround for https://github.com/pytorch/pytorch/issues/80800 
        CondFunc('torch.nn.functional.layer_norm', lambda orig_func, *args, **kwargs: orig_func(*([args[0].contiguous()] + list(args[1:])), **kwargs),
                                                                                        lambda _, *args, **kwargs: args and isinstance(args[0], torch.Tensor) and args[0].device.type == 'mps')
        # MPS workaround for https://github.com/pytorch/pytorch/issues/90532
        CondFunc('torch.Tensor.numpy', lambda orig_func, self, *args, **kwargs: orig_func(self.detach(), *args, **kwargs), lambda _, self, *args, **kwargs: self.requires_grad)
    elif version.parse(torch.__version__) > version.parse("1.13.1"):
        cumsum_needs_int_fix = not torch.Tensor([1,2]).to(torch.device("mps")).equal(torch.ShortTensor([1,1]).to(torch.device("mps")).cumsum(0))
        cumsum_needs_bool_fix = not torch.BoolTensor([True,True]).to(device=torch.device("mps"), dtype=torch.int64).equal(torch.BoolTensor([True,False]).to(torch.device("mps")).cumsum(0))
        cumsum_fix_func = lambda orig_func, input, *args, **kwargs: cumsum_fix(input, orig_func, *args, **kwargs)
        CondFunc('torch.cumsum', cumsum_fix_func, None)
        CondFunc('torch.Tensor.cumsum', cumsum_fix_func, None)
        CondFunc('torch.narrow', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).clone(), None)

