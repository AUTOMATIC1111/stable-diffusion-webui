import importlib
import torch

from modules import shared


def check_for_npu():
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu
    torch_npu.npu.set_device(0)

    try:
        # Will raise a RuntimeError if no NPU is found
        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


def get_npu_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"npu:{shared.cmd_opts.device_id}"
    return "npu:0"


def torch_npu_gc():
    # Work around due to bug in torch_npu, revert me after fixed, @see https://gitee.com/ascend/pytorch/issues/I8KECW?from=project-issue
    torch.npu.set_device(0)
    with torch.npu.device(get_npu_device_string()):
        torch.npu.empty_cache()


has_npu = check_for_npu()
