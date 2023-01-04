import sys, os, shlex
import contextlib
import torch
from modules import errors
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


def get_optimal_device():
    if torch.cuda.is_available():
        return torch.device(get_cuda_device_string())

    if has_mps():
        return torch.device("mps")

    return cpu


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


# MPS workaround for https://github.com/pytorch/pytorch/issues/79383
orig_tensor_to = torch.Tensor.to
def tensor_to_fix(self, *args, **kwargs):
    if self.device.type != 'mps' and \
       ((len(args) > 0 and isinstance(args[0], torch.device) and args[0].type == 'mps') or \
       (isinstance(kwargs.get('device'), torch.device) and kwargs['device'].type == 'mps')):
        self = self.contiguous()
    return orig_tensor_to(self, *args, **kwargs)


# MPS workaround for https://github.com/pytorch/pytorch/issues/80800 
orig_layer_norm = torch.nn.functional.layer_norm
def layer_norm_fix(*args, **kwargs):
    if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].device.type == 'mps':
        args = list(args)
        args[0] = args[0].contiguous()
    return orig_layer_norm(*args, **kwargs)


# MPS workaround for https://github.com/pytorch/pytorch/issues/90532
orig_tensor_numpy = torch.Tensor.numpy
def numpy_fix(self, *args, **kwargs):
    if self.requires_grad:
        self = self.detach()
    return orig_tensor_numpy(self, *args, **kwargs)


# PyTorch 1.13 doesn't need these fixes but unfortunately is slower and has regressions that prevent training from working
if has_mps() and version.parse(torch.__version__) < version.parse("1.13"):
    torch.Tensor.to = tensor_to_fix
    torch.nn.functional.layer_norm = layer_norm_fix
    torch.Tensor.numpy = numpy_fix
