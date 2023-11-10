import contextlib
from modules import shared
from modules.sd_hijack_utils import CondFunc

has_ipex = False
try:
    import torch
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except Exception:
    pass

def check_for_xpu():
    if not has_ipex:
        return False

    return hasattr(torch, 'xpu') and torch.xpu.is_available()

has_xpu = check_for_xpu()

def get_xpu_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"xpu:{shared.cmd_opts.device_id}"
    return "xpu"

def return_null_context(*args, **kwargs): # pylint: disable=unused-argument
    return contextlib.nullcontext()

if has_xpu:
    CondFunc('torch.Generator',
        lambda orig_func, device=None: torch.xpu.Generator(device),
        lambda orig_func, device=None: device is not None and device != torch.device("cpu") and device != "cpu")

    CondFunc('torch.nn.functional.layer_norm',
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        orig_func(input.to(weight.data.dtype), normalized_shape, weight, *args, **kwargs),
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        weight is not None and input.dtype != weight.data.dtype)

    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
