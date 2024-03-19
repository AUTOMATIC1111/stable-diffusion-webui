import importlib
import torch

from modules import shared


def check_for_npu():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu

    try:
        # Will raise a RuntimeError if no NPU is found
        _ = torch_npu.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


def get_npu_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"npu:{shared.cmd_opts.device_id}"
    return "npu:0"


def torch_npu_gc():
    with torch.npu.device(get_npu_device_string()):
        torch.npu.empty_cache()


has_npu = check_for_npu()
