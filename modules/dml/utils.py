from typing import Optional
import torch

rDevice = torch.device | int
def get_device(device: Optional[rDevice]=None) -> torch.device:
    if device is None:
        device = torch.dml.context_device or torch.dml.current_device()
    return torch.device(device)
