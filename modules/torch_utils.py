from __future__ import annotations

import torch.nn


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
