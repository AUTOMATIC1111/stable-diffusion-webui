import torch

def directml_init():
    from modules.dml.backend import DirectML # pylint: disable=ungrouped-imports
    # Alternative of torch.cuda for DirectML.
    torch.dml = DirectML

    torch.cuda.is_available = lambda: False
    torch.cuda.device = torch.dml.device
    torch.cuda.current_device = torch.dml.current_device
    torch.cuda.get_device_name = torch.dml.get_device_name
    torch.cuda.get_device_properties = torch.dml.get_device_properties

    torch.cuda.memory_stats = torch.dml.memory_stats
    torch.cuda.mem_get_info = torch.dml.mem_get_info
    torch.cuda.memory_allocated = torch.dml.memory_allocated
    torch.cuda.max_memory_allocated = torch.dml.max_memory_allocated
    torch.cuda.reset_peak_memory_stats = torch.dml.reset_peak_memory_stats

def directml_hijack_init():
    import modules.dml.hijack

def directml_override_opts():
    from modules import shared
    if shared.backend == shared.Backend.DIFFUSERS:
        shared.opts.diffusers_generator_device = "cpu" # DirectML does not support torch.Generator API.
