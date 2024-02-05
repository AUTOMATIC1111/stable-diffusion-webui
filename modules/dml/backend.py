# pylint: disable=no-member,no-self-argument,no-method-argument
from typing import Optional, Callable
import torch
import torch_directml # pylint: disable=import-error
import modules.dml.amp as amp
from .utils import rDevice, get_device
from .device import Device
from .Generator import Generator
from .device_properties import DeviceProperties


def amd_mem_get_info(device: Optional[rDevice]=None) -> tuple[int, int]:
    from .memory_amd import AMDMemoryProvider
    return AMDMemoryProvider.mem_get_info(get_device(device).index)


def pdh_mem_get_info(device: Optional[rDevice]=None) -> tuple[int, int]:
    mem_info = DirectML.memory_provider.get_memory(get_device(device).index)
    return (mem_info["total_committed"] - mem_info["dedicated_usage"], mem_info["total_committed"])


def mem_get_info(device: Optional[rDevice]=None) -> tuple[int, int]: # pylint: disable=unused-argument
    return (8589934592, 8589934592)


class DirectML:
    amp = amp
    device = Device
    Generator = Generator

    context_device: Optional[torch.device] = None

    is_autocast_enabled = False
    autocast_gpu_dtype = torch.float16

    memory_provider = None

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
        return {
            "num_ooms": 0,
            "num_alloc_retries": 0,
        }

    mem_get_info: Callable = mem_get_info

    def memory_allocated(device: Optional[rDevice]=None) -> int:
        return sum(torch_directml.gpu_memory(get_device(device).index)) * (1 << 20)

    def max_memory_allocated(device: Optional[rDevice]=None):
        return DirectML.memory_allocated(device) # DirectML does not empty GPU memory

    def reset_peak_memory_stats(device: Optional[rDevice]=None):
        return
