from typing import Optional
import torch


class Generator(torch.Generator):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__("cpu")
