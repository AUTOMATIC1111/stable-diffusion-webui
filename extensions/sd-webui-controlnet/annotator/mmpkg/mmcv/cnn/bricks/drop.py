# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from annotator.mmpkg.mmcv import build_from_cfg
from .registry import DROPOUT_LAYERS


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


@DROPOUT_LAYERS.register_module()
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@DROPOUT_LAYERS.register_module()
class Dropout(nn.Dropout):
    """A wrapper for ``torch.nn.Dropout``, We rename the ``p`` of
    ``torch.nn.Dropout`` to ``drop_prob`` so as to be consistent with
    ``DropPath``

    Args:
        drop_prob (float): Probability of the elements to be
            zeroed. Default: 0.5.
        inplace (bool):  Do the operation inplace or not. Default: False.
    """

    def __init__(self, drop_prob=0.5, inplace=False):
        super().__init__(p=drop_prob, inplace=inplace)


def build_dropout(cfg, default_args=None):
    """Builder for drop out layers."""
    return build_from_cfg(cfg, DROPOUT_LAYERS, default_args)
