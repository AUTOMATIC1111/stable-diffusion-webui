""" Activations (jit)

A collection of jit-scripted activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

All jit scripted activations are lacking in-place variations on purpose, scripted kernel fusion does not
currently work across in-place op boundaries, thus performance is equal to or less than the non-scripted
versions if they contain in-place ops.

Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F

__all__ = ['swish_jit', 'SwishJit', 'mish_jit', 'MishJit',
           'hard_sigmoid_jit', 'HardSigmoidJit', 'hard_swish_jit', 'HardSwishJit']


@torch.jit.script
def swish_jit(x, inplace: bool = False):
    """Swish - Described originally as SiLU (https://arxiv.org/abs/1702.03118v3)
    and also as Swish (https://arxiv.org/abs/1710.05941).

    TODO Rename to SiLU with addition to PyTorch
    """
    return x.mul(x.sigmoid())


@torch.jit.script
def mish_jit(x, _inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class SwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishJit, self).__init__()

    def forward(self, x):
        return swish_jit(x)


class MishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(MishJit, self).__init__()

    def forward(self, x):
        return mish_jit(x)


@torch.jit.script
def hard_sigmoid_jit(x, inplace: bool = False):
    # return F.relu6(x + 3.) / 6.
    return (x + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSigmoidJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoidJit, self).__init__()

    def forward(self, x):
        return hard_sigmoid_jit(x)


@torch.jit.script
def hard_swish_jit(x, inplace: bool = False):
    # return x * (F.relu6(x + 3.) / 6)
    return x * (x + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwishJit, self).__init__()

    def forward(self, x):
        return hard_swish_jit(x)
