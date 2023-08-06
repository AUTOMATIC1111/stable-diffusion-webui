import os
import contextlib
import torch
import intel_extension_for_pytorch as ipex
from modules import shared
from modules.sd_hijack_utils import CondFunc
from .diffusers import ipex_diffusers

#ControlNet depth_leres++
class DummyDataParallel(torch.nn.Module):
    def __new__(cls, module, device_ids=None, output_device=None, dim=0):
        if type(device_ids) is list and len(device_ids) > 1:
            shared.log.warning("IPEX backend doesn't support DataParallel on multiple XPU devices")
        return module.to(shared.device)

def ipex_no_cuda(orig_func, *args, **kwargs): # pylint: disable=redefined-outer-name
    torch.cuda.is_available = lambda: False
    orig_func(*args, **kwargs)
    torch.cuda.is_available = torch.xpu.is_available

def return_null_context(*args, **kwargs):
    return contextlib.nullcontext()

def ipex_init():
    #Replace cuda with xpu:
    torch.cuda.current_device = torch.xpu.current_device
    torch.cuda.current_stream = torch.xpu.current_stream
    torch.cuda.device = torch.xpu.device
    torch.cuda.device_count = torch.xpu.device_count
    torch.cuda.device_of = torch.xpu.device_of
    torch.cuda.getDeviceIdListForCard = torch.xpu.getDeviceIdListForCard
    torch.cuda.get_device_name = torch.xpu.get_device_name
    torch.cuda.get_device_properties = torch.xpu.get_device_properties
    torch.cuda.init = torch.xpu.init
    torch.cuda.is_available = torch.xpu.is_available
    torch.cuda.is_initialized = torch.xpu.is_initialized
    torch.cuda.set_device = torch.xpu.set_device
    torch.cuda.stream = torch.xpu.stream
    torch.cuda.synchronize = torch.xpu.synchronize
    torch.cuda.Event = torch.xpu.Event
    torch.cuda.Stream = torch.xpu.Stream
    torch.cuda.FloatTensor = torch.xpu.FloatTensor
    torch.Tensor.cuda = torch.Tensor.xpu
    torch.Tensor.is_cuda = torch.Tensor.is_xpu

    #Memory:
    torch.cuda.empty_cache = torch.xpu.empty_cache
    torch.cuda.memory_stats = torch.xpu.memory_stats
    torch.cuda.memory_summary = torch.xpu.memory_summary
    torch.cuda.memory_snapshot = torch.xpu.memory_snapshot
    torch.cuda.memory_allocated = torch.xpu.memory_allocated
    torch.cuda.max_memory_allocated = torch.xpu.max_memory_allocated
    torch.cuda.memory_reserved = torch.xpu.memory_reserved
    torch.cuda.max_memory_reserved = torch.xpu.max_memory_reserved
    torch.cuda.reset_peak_memory_stats = torch.xpu.reset_peak_memory_stats
    torch.cuda.memory_stats_as_nested_dict = torch.xpu.memory_stats_as_nested_dict
    torch.cuda.reset_accumulated_memory_stats = torch.xpu.reset_accumulated_memory_stats

    #RNG:
    torch.cuda.get_rng_state = torch.xpu.get_rng_state
    torch.cuda.get_rng_state_all = torch.xpu.get_rng_state_all
    torch.cuda.set_rng_state = torch.xpu.set_rng_state
    torch.cuda.set_rng_state_all = torch.xpu.set_rng_state_all
    torch.cuda.manual_seed = torch.xpu.manual_seed
    torch.cuda.manual_seed_all = torch.xpu.manual_seed_all
    torch.cuda.seed = torch.xpu.seed
    torch.cuda.seed_all = torch.xpu.seed_all
    torch.cuda.initial_seed = torch.xpu.initial_seed

    #Training:
    try:
        torch.cuda.amp.GradScaler = torch.xpu.amp.GradScaler
    except Exception:
        torch.cuda.amp.GradScaler = ipex.cpu.autocast._grad_scaler.GradScaler

    #C
    torch._C._cuda_getCurrentRawStream = ipex._C._getCurrentStream
    ipex._C._DeviceProperties.major = 2023
    ipex._C._DeviceProperties.minor = 2

    #Fix functions with ipex:
    torch.cuda.mem_get_info = lambda device=None: [(torch.xpu.get_device_properties(device).total_memory - torch.xpu.memory_allocated(device)), torch.xpu.get_device_properties(device).total_memory]
    torch._utils._get_available_device_type = lambda: "xpu" # pylint: disable=protected-access
    torch.xpu.empty_cache = torch.xpu.empty_cache if "WSL2" not in os.popen("uname -a").read() else lambda: None
    torch.cuda.get_device_properties.major = 2023
    torch.cuda.get_device_properties.minor = 2
    torch.backends.cuda.sdp_kernel = return_null_context
    torch.nn.DataParallel = DummyDataParallel
    torch.cuda.ipc_collect = lambda: None
    torch.cuda.utilization = lambda: 0

    #Libraries that blindly uses cuda:
    #Adetailer:
    CondFunc('torch.Tensor.to',
        lambda orig_func, self, device=None, *args, **kwargs: orig_func(self, shared.device, *args, **kwargs),
        lambda orig_func, self, device=None, *args, **kwargs: (type(device) is torch.device and device.type == "cuda") or (type(device) is str and "cuda" in device))
    CondFunc('torch.empty',
        lambda orig_func, *args, device=None, **kwargs: orig_func(*args, device=shared.device, **kwargs),
        lambda orig_func, *args, device=None, **kwargs: (type(device) is torch.device and device.type == "cuda") or (type(device) is str and "cuda" in device))
    #ControlNet depth_leres
    CondFunc('torch.load',
        lambda orig_func, f, map_location=None, *args, **kwargs: orig_func(f, shared.device, *args, **kwargs),
        lambda orig_func, f, map_location=None, *args, **kwargs: (map_location is None) or (type(map_location) is torch.device and map_location.type == "cuda") or (type(map_location) is str and "cuda" in map_location))

    #Broken functions when torch.cuda.is_available is True:
    #Pin Memory:
    CondFunc('torch.utils.data.dataloader._BaseDataLoaderIter.__init__',
        lambda orig_func, *args, **kwargs: ipex_no_cuda(orig_func, *args, **kwargs),
        lambda orig_func, *args, **kwargs: True)

    #Functions with dtype errors:
    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    #FP32:
    CondFunc('torch.nn.modules.Linear.forward',
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
        input.dtype != weight.data.dtype and weight is not None)
    #Embedding BF16
    CondFunc('torch.cat',
        lambda orig_func, input, *args, **kwargs: orig_func([input[0].to(input[1].dtype), input[1], input[2].to(input[1].dtype)], *args, **kwargs),
        lambda orig_func, input, *args, **kwargs: len(input) == 3 and (input[0].dtype != input[1].dtype or input[2].dtype != input[1].dtype))
    #Diffusers BF16:
    CondFunc('torch.nn.functional.conv2d',
        lambda orig_func, input, weight, *args, **kwargs: orig_func(input.to(weight.data.dtype), weight, *args, **kwargs),
        lambda orig_func, input, weight, *args, **kwargs: input.dtype != weight.data.dtype)

    #Functions that does not work with the XPU:
    #UniPC:
    CondFunc('torch.linalg.solve',
        lambda orig_func, A, B, *args, **kwargs: orig_func(A.to("cpu"), B.to("cpu"), *args, **kwargs).to(shared.device),
        lambda orig_func, A, B, *args, **kwargs: A.device != torch.device("cpu") or B.device != torch.device("cpu"))
    #SDE Samplers:
    CondFunc('torch.Generator',
        lambda orig_func, device: torch.xpu.Generator(device),
        lambda orig_func, device: device != torch.device("cpu") and device != "cpu")
    #Latent antialias:
    CondFunc('torch.nn.functional.interpolate',
        lambda orig_func, input, *args, **kwargs: orig_func(input.to("cpu"), *args, **kwargs).to(shared.device),
        lambda orig_func, input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False: antialias)
    #Diffusers Float64 (ARC GPUs doesn't support double or Float64):
    if not torch.xpu.has_fp64_dtype():
        CondFunc('torch.from_numpy',
        lambda orig_func, ndarray: orig_func(ndarray.astype('float32')),
        lambda orig_func, ndarray: ndarray.dtype == float)
    #ControlNet and TiledVAE:
    CondFunc('torch.batch_norm',
        lambda orig_func, input, weight, bias, *args, **kwargs: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=shared.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=shared.device), *args, **kwargs),
        lambda orig_func, input, *args, **kwargs: input.device != torch.device("cpu"))
    #ControlNet
    CondFunc('torch.instance_norm',
        lambda orig_func, input, weight, bias, *args, **kwargs: orig_func(input,
        weight if weight is not None else torch.ones(input.size()[1], device=shared.device),
        bias if bias is not None else torch.zeros(input.size()[1], device=shared.device), *args, **kwargs),
        lambda orig_func, input, *args, **kwargs: input.device != torch.device("cpu"))

    ipex_diffusers()
