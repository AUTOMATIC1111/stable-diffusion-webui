"""Modified from https://github.com/rwightman/pytorch-image-
models/blob/master/timm/models/layers/drop.py."""

import torch
from torch import nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    Args:
        drop_prob (float): Drop rate for paths of model. Dropout rate has
            to be between 0 and 1. Default: 0.
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        shape = (x.shape[0], ) + (1, ) * (
            x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output
