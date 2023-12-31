import contextlib
import torch
import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
from modules.sd_hijack_utils import CondFunc
from modules import devices

# pylint: disable=protected-access, missing-function-docstring, line-too-long, unnecessary-lambda, no-else-return

class DummyDataParallel(torch.nn.Module): # pylint: disable=missing-class-docstring, unused-argument, too-few-public-methods
    def __new__(cls, module, device_ids=None, output_device=None, dim=0): # pylint: disable=unused-argument
        if isinstance(device_ids, list) and len(device_ids) > 1:
            print("IPEX backend doesn't support DataParallel on multiple XPU devices")
        return module.to(devices.device)

def return_null_context(*args, **kwargs): # pylint: disable=unused-argument
    return contextlib.nullcontext()

@property
def is_cuda(self):
    return self.device.type == 'xpu'

def check_device(device):
    return bool((isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and "cuda" in device) or isinstance(device, int))

def return_xpu(device):
    return f"xpu:{device.split(':')[-1]}" if isinstance(device, str) and ":" in device else f"xpu:{device}" if isinstance(device, int) else torch.device(devices.device) if isinstance(device, torch.device) else devices.device


# Autocast
original_autocast = torch.autocast
def ipex_autocast(*args, **kwargs):
    if len(args) > 0 and args[0] == "cuda" or args[0] == "xpu":
        if "dtype" in kwargs:
            return original_autocast("xpu", *args[1:], **kwargs)
        else:
            return original_autocast("xpu", *args[1:], dtype=devices.dtype, **kwargs)
    else:
        return original_autocast(*args, **kwargs)

# Latent Antialias CPU Offload:
original_interpolate = torch.nn.functional.interpolate
def interpolate(tensor, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False): # pylint: disable=too-many-arguments
    if antialias or align_corners is not None:
        return_device = tensor.device
        return_dtype = tensor.dtype
        return original_interpolate(tensor.to("cpu", dtype=torch.float32), size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias).to(return_device, dtype=return_dtype)
    else:
        return original_interpolate(tensor, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)

# Diffusers Float64 (Alchemist GPUs doesn't support 64 bit):
original_from_numpy = torch.from_numpy
def from_numpy(ndarray):
    if ndarray.dtype == float:
        return original_from_numpy(ndarray.astype('float32'))
    else:
        return original_from_numpy(ndarray)

if torch.xpu.has_fp64_dtype():
    original_torch_bmm = torch.bmm
    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
else:
    # 32 bit attention workarounds for Alchemist:
    try:
        from .attention import torch_bmm_32_bit as original_torch_bmm
        from .attention import scaled_dot_product_attention_32_bit as original_scaled_dot_product_attention
    except Exception: # pylint: disable=broad-exception-caught
        original_torch_bmm = torch.bmm
        original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention


# Data Type Errors:
def torch_bmm(input, mat2, *, out=None):
    if input.dtype != mat2.dtype:
        mat2 = mat2.to(input.dtype)
    return original_torch_bmm(input, mat2, out=out)

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    if query.dtype != key.dtype:
        key = key.to(dtype=query.dtype)
    if query.dtype != value.dtype:
        value = value.to(dtype=query.dtype)
    return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

# A1111 FP16
original_functional_group_norm = torch.nn.functional.group_norm
def functional_group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    if weight is not None and input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and weight is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)

# A1111 BF16
original_functional_layer_norm = torch.nn.functional.layer_norm
def functional_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    if weight is not None and input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and weight is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

# Training
original_functional_linear = torch.nn.functional.linear
def functional_linear(input, weight, bias=None):
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_linear(input, weight, bias=bias)

original_functional_conv2d = torch.nn.functional.conv2d
def functional_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

# A1111 Embedding BF16
original_torch_cat = torch.cat
def torch_cat(tensor, *args, **kwargs):
    if len(tensor) == 3 and (tensor[0].dtype != tensor[1].dtype or tensor[2].dtype != tensor[1].dtype):
        return original_torch_cat([tensor[0].to(tensor[1].dtype), tensor[1], tensor[2].to(tensor[1].dtype)], *args, **kwargs)
    else:
        return original_torch_cat(tensor, *args, **kwargs)

# SwinIR BF16:
original_funtional_pad = torch.nn.functional.pad
def funtional_pad(input, pad, mode='constant', value=None):
    if mode == 'reflect' and input.dtype == torch.bfloat16:
        return original_funtional_pad(input.to(torch.float32), pad, mode=mode, value=value).to(dtype=torch.bfloat16)
    else:
        return original_funtional_pad(input, pad, mode=mode, value=value)


def ipex_hijacks():
    CondFunc('torch.tensor',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.Tensor.to',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, return_xpu(device), *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.Tensor.cuda',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, return_xpu(device), *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.UntypedStorage.__init__',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.UntypedStorage.cuda',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, return_xpu(device), *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.empty',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.randn',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.ones',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.zeros',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.linspace',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=return_xpu(device), **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.Generator',
        lambda orig_func, device=None: orig_func(return_xpu(device)),
        lambda orig_func, device=None: check_device(device))
    CondFunc('torch.load',
        lambda orig_func, f, map_location=None, pickle_module=None, *, weights_only=False, mmap=None, **kwargs:
        orig_func(f, map_location=return_xpu(map_location), pickle_module=pickle_module, weights_only=weights_only, mmap=mmap, **kwargs),
        lambda orig_func, f, map_location=None, pickle_module=None, *, weights_only=False, mmap=None, **kwargs: check_device(map_location))

    # Hijack Functions:
    torch.backends.cuda.sdp_kernel = return_null_context
    torch.nn.DataParallel = DummyDataParallel
    torch.UntypedStorage.is_cuda = is_cuda
    torch.autocast = ipex_autocast

    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    torch.nn.functional.group_norm = functional_group_norm
    torch.nn.functional.layer_norm = functional_layer_norm
    torch.nn.functional.linear = functional_linear
    torch.nn.functional.conv2d = functional_conv2d
    torch.nn.functional.interpolate = interpolate
    torch.nn.functional.pad = funtional_pad

    torch.bmm = torch_bmm
    torch.cat = torch_cat
    if not torch.xpu.has_fp64_dtype():
        torch.from_numpy = from_numpy
