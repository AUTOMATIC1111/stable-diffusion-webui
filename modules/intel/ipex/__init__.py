import os
import sys
import contextlib
import torch
import intel_extension_for_pytorch as ipex
from modules import shared
from .diffusers import ipex_diffusers
from .hijacks import ipex_hijacks

#ControlNet depth_leres++
class DummyDataParallel(torch.nn.Module):
    def __new__(cls, module, device_ids=None, output_device=None, dim=0):
        if type(device_ids) is list and len(device_ids) > 1:
            shared.log.warning("IPEX backend doesn't support DataParallel on multiple XPU devices")
        return module.to(shared.device)

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
    torch.cuda._initialization_lock = torch.xpu.lazy_init._initialization_lock
    torch.cuda._initialized = torch.xpu.lazy_init._initialized
    torch.cuda._lazy_seed_tracker = torch.xpu.lazy_init._lazy_seed_tracker
    torch.cuda._queued_calls = torch.xpu.lazy_init._queued_calls
    torch.cuda._tls = torch.xpu.lazy_init._tls
    torch.cuda.threading = torch.xpu.lazy_init.threading
    torch.cuda.traceback = torch.xpu.lazy_init.traceback
    torch.cuda.Optional = torch.xpu.Optional
    torch.cuda.__cached__ = torch.xpu.__cached__
    torch.cuda.__loader__ = torch.xpu.__loader__
    torch.cuda.ComplexFloatStorage = torch.xpu.ComplexFloatStorage
    torch.cuda.Tuple = torch.xpu.Tuple
    torch.cuda.streams = torch.xpu.streams
    torch.cuda._lazy_new = torch.xpu._lazy_new
    torch.cuda.FloatStorage = torch.xpu.FloatStorage
    torch.cuda.Any = torch.xpu.Any
    torch.cuda.__doc__ = torch.xpu.__doc__
    torch.cuda.default_generators = torch.xpu.default_generators
    torch.cuda.HalfTensor = torch.xpu.HalfTensor
    torch.cuda._get_device_index = torch.xpu._get_device_index
    torch.cuda.__path__ = torch.xpu.__path__
    torch.cuda.Device = torch.xpu.Device
    torch.cuda.IntTensor = torch.xpu.IntTensor
    torch.cuda.ByteStorage = torch.xpu.ByteStorage
    torch.cuda.set_stream = torch.xpu.set_stream
    torch.cuda.BoolStorage = torch.xpu.BoolStorage
    torch.cuda.get_device_capability = torch.xpu.get_device_capability
    torch.cuda.os = torch.xpu.os
    torch.cuda.torch = torch.xpu.torch
    torch.cuda.BFloat16Storage = torch.xpu.BFloat16Storage
    torch.cuda.Union = torch.xpu.Union
    torch.cuda.DoubleTensor = torch.xpu.DoubleTensor
    torch.cuda.ShortTensor = torch.xpu.ShortTensor
    torch.cuda.LongTensor = torch.xpu.LongTensor
    torch.cuda.IntStorage = torch.xpu.IntStorage
    torch.cuda.LongStorage = torch.xpu.LongStorage
    torch.cuda.__annotations__ = torch.xpu.__annotations__
    torch.cuda.__package__ = torch.xpu.__package__
    torch.cuda.__builtins__ = torch.xpu.__builtins__
    torch.cuda.CharTensor = torch.xpu.CharTensor
    torch.cuda.List = torch.xpu.List
    torch.cuda._lazy_init = torch.xpu._lazy_init
    torch.cuda.BFloat16Tensor = torch.xpu.BFloat16Tensor
    torch.cuda.DoubleStorage = torch.xpu.DoubleStorage
    torch.cuda.ByteTensor = torch.xpu.ByteTensor
    torch.cuda.StreamContext = torch.xpu.StreamContext
    torch.cuda.ComplexDoubleStorage = torch.xpu.ComplexDoubleStorage
    torch.cuda.ShortStorage = torch.xpu.ShortStorage
    torch.cuda._lazy_call = torch.xpu._lazy_call
    torch.cuda.HalfStorage = torch.xpu.HalfStorage
    torch.cuda.random = torch.xpu.random
    torch.cuda._device = torch.xpu._device
    torch.cuda.classproperty = torch.xpu.classproperty
    torch.cuda.__name__ = torch.xpu.__name__
    torch.cuda._device_t = torch.xpu._device_t
    torch.cuda.warnings = torch.xpu.warnings
    torch.cuda.__spec__ = torch.xpu.__spec__
    torch.cuda.BoolTensor = torch.xpu.BoolTensor
    torch.cuda.CharStorage = torch.xpu.CharStorage
    torch.cuda.__file__ = torch.xpu.__file__
    torch.cuda._is_in_bad_fork = torch.xpu.lazy_init._is_in_bad_fork
    #torch.cuda.is_current_stream_capturing = torch.xpu.is_current_stream_capturing

    #Memory:
    torch.cuda.memory = torch.xpu.memory
    if 'linux' in sys.platform and "WSL2" in os.popen("uname -a").read():
        torch.xpu.empty_cache = lambda: None
    torch.cuda.empty_cache = torch.xpu.empty_cache
    torch.cuda.memory_stats = torch.xpu.memory_stats
    torch.cuda.memory_summary = torch.xpu.memory_summary
    torch.cuda.memory_snapshot = torch.xpu.memory_snapshot
    torch.cuda.memory_allocated = torch.xpu.memory_allocated
    torch.cuda.max_memory_allocated = torch.xpu.max_memory_allocated
    torch.cuda.memory_reserved = torch.xpu.memory_reserved
    torch.cuda.memory_cached = torch.xpu.memory_reserved
    torch.cuda.max_memory_reserved = torch.xpu.max_memory_reserved
    torch.cuda.max_memory_cached = torch.xpu.max_memory_reserved
    torch.cuda.reset_peak_memory_stats = torch.xpu.reset_peak_memory_stats
    torch.cuda.reset_max_memory_cached = torch.xpu.reset_peak_memory_stats
    torch.cuda.reset_max_memory_allocated = torch.xpu.reset_peak_memory_stats
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

    #AMP:
    torch.cuda.amp = torch.xpu.amp
    if not hasattr(torch.cuda.amp, "common"):
        torch.cuda.amp.common = contextlib.nullcontext()
    torch.cuda.amp.common.amp_definitely_not_available = lambda: False
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
    torch.has_cuda = True
    torch.cuda.has_half = True
    torch.cuda.is_bf16_supported = True
    #torch.version.cuda = "11.7" #Breaks System Info
    torch.cuda.get_device_properties.major = 11
    torch.cuda.get_device_properties.minor = 7
    torch.backends.cuda.sdp_kernel = return_null_context
    torch.nn.DataParallel = DummyDataParallel
    torch.cuda.ipc_collect = lambda: None
    torch.cuda.utilization = lambda: 0

    ipex_hijacks()
    ipex_diffusers()
