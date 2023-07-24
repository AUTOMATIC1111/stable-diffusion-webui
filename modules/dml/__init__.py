import os
import torch

from modules.sd_hijack_utils import CondFunc

do_nothing = lambda: None

def directml_init():
    from modules.dml.backend import DirectML # pylint: disable=ungrouped-imports
    # Alternative of torch.cuda for DirectML.
    torch.dml = DirectML

    torch.cuda.is_available = lambda: False
    torch.cuda.device = torch.dml.device
    torch.cuda.device_count = torch.dml.device_count
    torch.cuda.current_device = torch.dml.current_device
    torch.cuda.get_device_name = torch.dml.get_device_name
    torch.cuda.get_device_properties = torch.dml.get_device_properties

    torch.cuda.empty_cache = do_nothing
    torch.cuda.ipc_collect = do_nothing
    torch.cuda.memory_stats = torch.dml.memory_stats
    torch.cuda.mem_get_info = torch.dml.mem_get_info
    torch.cuda.memory_allocated = torch.dml.memory_allocated
    torch.cuda.max_memory_allocated = torch.dml.max_memory_allocated
    torch.cuda.reset_peak_memory_stats = torch.dml.reset_peak_memory_stats
    torch.cuda.utilization = lambda: 0

    torch.Tensor.directml = lambda self: self.to(torch.dml.current_device())

    mem_bound = os.environ.get("DML_GPU_MEMORY_BOUND", None)
    if mem_bound is not None:
        torch.dml.set_gpu_memory_bound(int(mem_bound))

def directml_do_hijack():
    import modules.dml.hijack
    from modules.devices import device

    if not torch.dml.has_float64_support(device):
        CondFunc('torch.from_numpy',
            lambda orig_func, *args, **kwargs: orig_func(args[0].astype('float32')),
            lambda *args, **kwargs: args[1].dtype == float)

def directml_override_opts():
    from modules import shared
    if shared.backend == shared.Backend.DIFFUSERS:
        shared.opts.diffusers_generator_device = "cpu" # DirectML does not support torch.Generator API.
