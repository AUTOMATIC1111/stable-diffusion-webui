# pylint: disable=no-member,no-self-argument,no-method-argument
from typing import Optional
import torch
import torch_directml # pylint: disable=import-error
import modules.dml.amp as amp

from .memctl.unknown import UnknownMemoryControl
from .utils import rDevice, get_device
from .device import device
from .device_properties import DeviceProperties

class DirectML:
    amp = amp
    device = device

    context_device: torch.device | None = None

    __gpu_memory_bound: int | None = None
    
    is_autocast_enabled = False
    autocast_gpu_dtype = torch.float16

    def __get_memory_control(device: torch.device):
        assert device.type == 'privateuseone'
        try:
            device_name = torch_directml.device_name(device.index)
            if 'NVIDIA' in device_name or 'GeForce' in device_name:
                from .memctl.nvidia import nVidiaMemoryControl as memory_control
            elif 'AMD' in device_name or 'Radeon' in device_name:
                from .memctl.amd import AMDMemoryControl as memory_control
            elif 'Intel' in device_name:
                from .memctl.intel import IntelMemoryControl as memory_control
            else:
                return UnknownMemoryControl
            return memory_control
        except Exception:
            return UnknownMemoryControl
        
    def set_gpu_memory_bound(bound: int | None):
        DirectML.__gpu_memory_bound = bound

    def is_available() -> bool:
        return torch_directml.is_available()

    def is_directml_device(device: torch.device) -> bool:
        return device.type == "privateuseone"

    def has_float64_support(device: Optional[rDevice]=None) -> bool:
        return torch_directml.has_float64_support(get_device(device).index)

    def device_count() -> int:
        return torch_directml.device_count()

    def current_device() -> torch.device:
        return DirectML.context_device or DirectML.default_device()

    def default_device() -> torch.device:
        return torch_directml.device(torch_directml.default_device())

    def get_device_string(device: Optional[rDevice]=None) -> str:
        return f"privateuseone:{get_device(device).index}"

    def get_device_name(device: Optional[rDevice]=None) -> str:
        return torch_directml.device_name(get_device(device).index)

    def get_device_properties(device: Optional[rDevice]=None) -> DeviceProperties:
        return DeviceProperties(get_device(device))

    def memory_stats(device: Optional[rDevice]=None):
        mem_stat_fill = "DirectMLDevice"
        return {
            "num_ooms": 0,
            "num_alloc_retries": mem_stat_fill,
        }

    def mem_get_info(device: Optional[rDevice]=None) -> tuple[int, int]:
        device = get_device(device)
        memory_control = DirectML.__get_memory_control(device)
        mem_info = memory_control.mem_get_info(device.index)
        if DirectML.__gpu_memory_bound is None:
            return mem_info
        used = mem_info[1] - mem_info[0]
        available = DirectML.__gpu_memory_bound - used
        return (0 if available < 0 else available, DirectML.__gpu_memory_bound)

    def memory_allocated(device: Optional[rDevice]=None) -> int:
        return sum(torch_directml.gpu_memory(get_device(device).index)) * (1 << 20)

    def max_memory_allocated(device: Optional[rDevice]=None):
        return DirectML.memory_allocated(device) # DirectML does not empty GPU memory

    def reset_peak_memory_stats(device: Optional[rDevice]=None):
        return
