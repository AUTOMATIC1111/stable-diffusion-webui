from typing import Optional
import torch

from .utils import rDevice, get_device

class device:
    def __enter__(self, device: Optional[rDevice]=None):
        torch.dml.context_device = get_device(device)

    def __init__(self, device: Optional[rDevice]=None) -> torch.device:
        return get_device(device)

    def __exit__(self, type, val, tb):
        torch.dml.context_device = None
