from typing import Optional, Union
import torch


rDevice = Union[torch.device, int]
def get_device(device: Optional[rDevice]=None) -> torch.device:
    if device is None:
        device = torch.dml.current_device()
    return torch.device(device)
