import torch
from typing import Optional

class Generator(torch.Generator):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__("cpu")
