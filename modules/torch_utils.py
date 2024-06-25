from __future__ import annotations

import torch.nn
import torch


def get_param(model) -> torch.nn.Parameter:
    """
    Find the first parameter in a model or module.
    """
    if hasattr(model, "model") and hasattr(model.model, "parameters"):
        # Unpeel a model descriptor to get at the actual Torch module.
        model = model.model

    for param in model.parameters():
        return param

    raise ValueError(f"No parameters found in model {model!r}")


def float64(t: torch.Tensor):
    """return torch.float64 if device is not mps or xpu, else return torch.float32"""
    if t.device.type in ['mps', 'xpu']:
        return torch.float32
    return torch.float64
