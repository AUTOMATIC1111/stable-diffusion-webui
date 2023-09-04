import torch
import intel_extension_for_pytorch as ipex
from modules import devices
from modules.sd_hijack_utils import CondFunc

def check_device(device):
    return bool((isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and "cuda" in device) or isinstance(device, int))

def ipex_no_cuda(orig_func, *args, **kwargs): # pylint: disable=redefined-outer-name
    torch.cuda.is_available = lambda: False
    orig_func(*args, **kwargs)
    torch.cuda.is_available = torch.xpu.is_available

original_autocast = torch.autocast
def ipex_autocast(*args, **kwargs):
    if args[0] == "cuda" or args[0] == "xpu":
        if "dtype" in kwargs:
            return original_autocast("xpu", *args[1:], **kwargs)
        else:
            return original_autocast("xpu", *args[1:], dtype=devices.dtype, **kwargs)
    else:
        return original_autocast(*args, **kwargs)

#Embedding BF16
original_torch_cat = torch.cat
def torch_cat(input, *args, **kwargs):
    if len(input) == 3 and (input[0].dtype != input[1].dtype or input[2].dtype != input[1].dtype):
        return original_torch_cat([input[0].to(input[1].dtype), input[1], input[2].to(input[1].dtype)], *args, **kwargs)
    else:
        return original_torch_cat(input, *args, **kwargs)

#Latent antialias:
original_interpolate = torch.nn.functional.interpolate
def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if antialias:
        return original_interpolate(input.to("cpu", dtype=torch.float32), size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias).to(devices.device, dtype=devices.dtype)
    else:
        return original_interpolate(input, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)

def ipex_hijacks():
    CondFunc('torch.Tensor.to',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, devices.device, *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.Tensor.cuda',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, devices.device, *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: check_device(device))
    CondFunc('torch.empty',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=devices.device, **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.load',
        lambda orig_func, *args, map_location=None, **kwargs: orig_func(*args, devices.device, **kwargs),
        lambda orig_func, *args, map_location=None, **kwargs: map_location is None or check_device(map_location))
    CondFunc('torch.randn',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=devices.device, **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.ones',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=devices.device, **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.zeros',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=devices.device, **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))
    CondFunc('torch.tensor',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=devices.device, **kwargs),
        lambda orig_func, *args, device=None, **kwargs: check_device(device))

    CondFunc('torch.Generator',
        lambda orig_func, device: torch.xpu.Generator(device),
        lambda orig_func, device: device != torch.device("cpu") and device != "cpu")
    #Crashes the GPU:
    CondFunc('torch.linalg.solve',
        lambda orig_func, A, B, *args, **kwargs: orig_func(A.to("cpu"), B.to("cpu"), *args, **kwargs).to(devices.device),
        lambda orig_func, A, B, *args, **kwargs: A.device != torch.device("cpu") or B.device != torch.device("cpu"))

    #TiledVAE and ControlNet:
    CondFunc('torch.batch_norm',
        lambda orig_func, input, weight, bias, *args, **kwargs: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=devices.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=devices.device), *args, **kwargs),
        lambda orig_func, input, *args, **kwargs: input.device != torch.device("cpu"))
    CondFunc('torch.instance_norm',
        lambda orig_func, input, weight, bias, *args, **kwargs: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=devices.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=devices.device), *args, **kwargs),
        lambda orig_func, input, *args, **kwargs: input.device != torch.device("cpu"))

    #Functions with dtype errors:
    #Original backend:
    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    #Embedding FP32:
    CondFunc('torch.bmm',
        lambda orig_func, input, mat2, *args, **kwargs: orig_func(input, mat2.to(input.dtype), *args, **kwargs),
        lambda orig_func, input, mat2, *args, **kwargs: input.dtype != mat2.dtype)
    #BF16:
    CondFunc('torch.nn.functional.layer_norm',
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        orig_func(input.to(weight.data.dtype), normalized_shape, weight, *args, **kwargs),
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        weight is not None and input.dtype != weight.data.dtype)

    #Diffusers Float64 (ARC GPUs doesn't support double or Float64):
    if not torch.xpu.has_fp64_dtype():
        CondFunc('torch.from_numpy',
        lambda orig_func, ndarray: orig_func(ndarray.astype('float32')),
        lambda orig_func, ndarray: ndarray.dtype == float)

    #Broken functions when torch.cuda.is_available is True:
    #Pin Memory:
    CondFunc('torch.utils.data.dataloader._BaseDataLoaderIter.__init__',
        lambda orig_func, *args, **kwargs: ipex_no_cuda(orig_func, *args, **kwargs),
        lambda orig_func, *args, **kwargs: True)

    #Functions that make compile mad with CondFunc:
    torch.autocast = ipex_autocast
    torch.cat = torch_cat
    torch.nn.functional.interpolate = interpolate
