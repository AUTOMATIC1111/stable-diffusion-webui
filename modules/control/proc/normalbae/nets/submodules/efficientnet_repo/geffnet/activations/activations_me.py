""" Activations (memory-efficient w/ custom autograd)

A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

These activations are not compatible with jit scripting or ONNX export of the model, please use either
the JIT or basic versions of the activations.

Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


__all__ = ['swish_me', 'SwishMe', 'mish_me', 'MishMe',
           'hard_sigmoid_me', 'HardSigmoidMe', 'hard_swish_me', 'HardSwishMe']


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish w/ memory-efficient checkpoint
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200

    Swish - Described originally as SiLU (https://arxiv.org/abs/1702.03118v3)
    and also as Swish (https://arxiv.org/abs/1710.05941).

    TODO Rename to SiLU with addition to PyTorch
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish_me(x, inplace=False):
    return SwishJitAutoFn.apply(x)


class SwishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishMe, self).__init__()

    def forward(self, x):
        return SwishJitAutoFn.apply(x)


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish_me(x, inplace=False):
    return MishJitAutoFn.apply(x)


class MishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(MishMe, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)


@torch.jit.script
def hard_sigmoid_jit_fwd(x, inplace: bool = False):
    return (x + 3).clamp(min=0, max=6).div(6.)


@torch.jit.script
def hard_sigmoid_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * ((x >= -3.) & (x <= 3.)) / 6.
    return grad_output * m


class HardSigmoidJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_sigmoid_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_sigmoid_jit_bwd(x, grad_output)


def hard_sigmoid_me(x, inplace: bool = False):
    return HardSigmoidJitAutoFn.apply(x)


class HardSigmoidMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoidMe, self).__init__()

    def forward(self, x):
        return HardSigmoidJitAutoFn.apply(x)


@torch.jit.script
def hard_swish_jit_fwd(x):
    return x * (x + 3).clamp(min=0, max=6).div(6.)


@torch.jit.script
def hard_swish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= 3.)
    m = torch.where((x >= -3.) & (x <= 3.),  x / 3. + .5, m)
    return grad_output * m


class HardSwishJitAutoFn(torch.autograd.Function):
    """A memory efficient, jit-scripted HardSwish activation"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_swish_jit_bwd(x, grad_output)


def hard_swish_me(x, inplace=False):
    return HardSwishJitAutoFn.apply(x)


class HardSwishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwishMe, self).__init__()

    def forward(self, x):
        return HardSwishJitAutoFn.apply(x)
