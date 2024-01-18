from typing import Optional, Union
import time
import torch
import torch.nn as nn
import accelerate.utils.modeling
from modules import devices


tensor_to_timer = 0
orig_method = accelerate.utils.set_module_tensor_to_device


def check_device_same(d1, d2):
    if d1.type != d2.type:
        return False
    if d1.type == "cuda" and d1.index is None:
        d1 = torch.device("cuda", index=0)
    if d2.type == "cuda" and d2.index is None:
        d2 = torch.device("cuda", index=0)
    return d1 == d2


# called for every item in state_dict by diffusers during model load
def hijack_set_module_tensor(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None, # pylint: disable=unused-argument
    fp16_statistics: Optional[torch.HalfTensor] = None, # pylint: disable=unused-argument
):
    global tensor_to_timer # pylint: disable=global-statement
    if device == 'cpu': # override to load directly to gpu
        device = devices.device
    t0 = time.time()
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            module = getattr(module, split)
        tensor_name = splits[-1]
    old_value = getattr(module, tensor_name)
    with devices.inference_context():
        # note: majority of time is spent on .to(old_value.dtype)
        if tensor_name in module._buffers: # pylint: disable=protected-access
            module._buffers[tensor_name] = value.to(device, old_value.dtype, non_blocking=True)  # pylint: disable=protected-access
        elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):  # pylint: disable=protected-access
            param_cls = type(module._parameters[tensor_name]) # pylint: disable=protected-access
            module._parameters[tensor_name] = param_cls(value, requires_grad=old_value.requires_grad).to(device, old_value.dtype, non_blocking=True) # pylint: disable=protected-access
    t1 = time.time()
    tensor_to_timer += (t1 - t0)


def hijack_accelerate():
    accelerate.utils.set_module_tensor_to_device = hijack_set_module_tensor
    global tensor_to_timer # pylint: disable=global-statement
    tensor_to_timer = 0


def restore_accelerate():
    accelerate.utils.set_module_tensor_to_device = orig_method
