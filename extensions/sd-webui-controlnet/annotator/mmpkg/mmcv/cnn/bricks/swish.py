# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from .registry import ACTIVATION_LAYERS


@ACTIVATION_LAYERS.register_module()
class Swish(nn.Module):
    """Swish Module.

    This module applies the swish function:

    .. math::
        Swish(x) = x * Sigmoid(x)

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
