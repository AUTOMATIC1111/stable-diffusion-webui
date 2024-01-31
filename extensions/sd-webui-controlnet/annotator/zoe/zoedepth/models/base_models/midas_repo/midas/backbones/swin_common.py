import torch

import torch.nn as nn
import numpy as np

from .utils import activations, forward_default, get_activation, Transpose


def forward_swin(pretrained, x):
    return forward_default(pretrained, x)


def _make_swin_backbone(
        model,
        hooks=[1, 1, 17, 1],
        patch_grid=[96, 96]
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.layers[0].blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.layers[1].blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.layers[2].blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.layers[3].blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if hasattr(model, "patch_grid"):
        used_patch_grid = model.patch_grid
    else:
        used_patch_grid = patch_grid

    patch_grid_size = np.array(used_patch_grid, dtype=int)

    pretrained.act_postprocess1 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size(patch_grid_size.tolist()))
    )
    pretrained.act_postprocess2 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 2).tolist()))
    )
    pretrained.act_postprocess3 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 4).tolist()))
    )
    pretrained.act_postprocess4 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 8).tolist()))
    )

    return pretrained
