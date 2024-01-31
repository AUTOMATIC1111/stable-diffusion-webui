# modified from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/fused_act.py # noqa:E501

# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator
# Augmentation (ADA)
# =======================================================================

# 1. Definitions

# "Licensor" means any person or entity that distributes its Work.

# "Software" means the original work of authorship made available under
# this License.

# "Work" means the Software and any additions to or derivative works of
# the Software that are made available under this License.

# The terms "reproduce," "reproduction," "derivative works," and
# "distribution" have the meaning as provided under U.S. copyright law;
# provided, however, that for the purposes of this License, derivative
# works shall not include works that remain separable from, or merely
# link (or bind by name) to the interfaces of, the Work.

# Works, including the Software, are "made available" under this License
# by including in or with the Work either (a) a copyright notice
# referencing the applicability of this License to the Work, or (b) a
# copy of this License.

# 2. License Grants

#     2.1 Copyright Grant. Subject to the terms and conditions of this
#     License, each Licensor grants to you a perpetual, worldwide,
#     non-exclusive, royalty-free, copyright license to reproduce,
#     prepare derivative works of, publicly display, publicly perform,
#     sublicense and distribute its Work and any resulting derivative
#     works in any form.

# 3. Limitations

#     3.1 Redistribution. You may reproduce or distribute the Work only
#     if (a) you do so under this License, (b) you include a complete
#     copy of this License with your distribution, and (c) you retain
#     without modification any copyright, patent, trademark, or
#     attribution notices that are present in the Work.

#     3.2 Derivative Works. You may specify that additional or different
#     terms apply to the use, reproduction, and distribution of your
#     derivative works of the Work ("Your Terms") only if (a) Your Terms
#     provide that the use limitation in Section 3.3 applies to your
#     derivative works, and (b) you identify the specific derivative
#     works that are subject to Your Terms. Notwithstanding Your Terms,
#     this License (including the redistribution requirements in Section
#     3.1) will continue to apply to the Work itself.

#     3.3 Use Limitation. The Work and any derivative works thereof only
#     may be used or intended for use non-commercially. Notwithstanding
#     the foregoing, NVIDIA and its affiliates may use the Work and any
#     derivative works commercially. As used herein, "non-commercially"
#     means for research or evaluation purposes only.

#     3.4 Patent Claims. If you bring or threaten to bring a patent claim
#     against any Licensor (including any claim, cross-claim or
#     counterclaim in a lawsuit) to enforce any patents that you allege
#     are infringed by any Work, then your rights under this License from
#     such Licensor (including the grant in Section 2.1) will terminate
#     immediately.

#     3.5 Trademarks. This License does not grant any rights to use any
#     Licensor’s or its affiliates’ names, logos, or trademarks, except
#     as necessary to reproduce the notices described in this License.

#     3.6 Termination. If you violate any term of this License, then your
#     rights under this License (including the grant in Section 2.1) will
#     terminate immediately.

# 4. Disclaimer of Warranty.

# THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
# NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
# THIS LICENSE.

# 5. Limitation of Liability.

# EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
# THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
# SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
# OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
# (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
# LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
# COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGES.

# =======================================================================

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['fused_bias_leakyrelu'])


class FusedBiasLeakyReLUFunctionBackward(Function):
    """Calculate second order deviation.

    This function is to compute the second order deviation for the fused leaky
    relu operation.
    """

    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = ext_module.fused_bias_leakyrelu(
            grad_output,
            empty,
            out,
            act=3,
            grad=1,
            alpha=negative_slope,
            scale=scale)

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors

        # The second order deviation, in fact, contains two parts, while the
        # the first part is zero. Thus, we direct consider the second part
        # which is similar with the first order deviation in implementation.
        gradgrad_out = ext_module.fused_bias_leakyrelu(
            gradgrad_input,
            gradgrad_bias.to(out.dtype),
            out,
            act=3,
            grad=1,
            alpha=ctx.negative_slope,
            scale=ctx.scale)

        return gradgrad_out, None, None, None


class FusedBiasLeakyReLUFunction(Function):

    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)

        out = ext_module.fused_bias_leakyrelu(
            input,
            bias,
            empty,
            act=3,
            grad=0,
            alpha=negative_slope,
            scale=scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedBiasLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale)

        return grad_input, grad_bias, None, None


class FusedBiasLeakyReLU(nn.Module):
    """Fused bias leaky ReLU.

    This function is introduced in the StyleGAN2:
    http://arxiv.org/abs/1912.04958

    The bias term comes from the convolution operation. In addition, to keep
    the variance of the feature map or gradients unchanged, they also adopt a
    scale similarly with Kaiming initialization. However, since the
    :math:`1+{alpha}^2` : is too small, we can just ignore it. Therefore, the
    final scale is just :math:`\sqrt{2}`:. Of course, you may change it with # noqa: W605, E501
    your own scale.

    TODO: Implement the CPU version.

    Args:
        channel (int): The channel number of the feature map.
        negative_slope (float, optional): Same as nn.LeakyRelu.
            Defaults to 0.2.
        scale (float, optional): A scalar to adjust the variance of the feature
            map. Defaults to 2**0.5.
    """

    def __init__(self, num_channels, negative_slope=0.2, scale=2**0.5):
        super(FusedBiasLeakyReLU, self).__init__()

        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_bias_leakyrelu(input, self.bias, self.negative_slope,
                                    self.scale)


def fused_bias_leakyrelu(input, bias, negative_slope=0.2, scale=2**0.5):
    """Fused bias leaky ReLU function.

    This function is introduced in the StyleGAN2:
    http://arxiv.org/abs/1912.04958

    The bias term comes from the convolution operation. In addition, to keep
    the variance of the feature map or gradients unchanged, they also adopt a
    scale similarly with Kaiming initialization. However, since the
    :math:`1+{alpha}^2` : is too small, we can just ignore it. Therefore, the
    final scale is just :math:`\sqrt{2}`:. Of course, you may change it with # noqa: W605, E501
    your own scale.

    Args:
        input (torch.Tensor): Input feature map.
        bias (nn.Parameter): The bias from convolution operation.
        negative_slope (float, optional): Same as nn.LeakyRelu.
            Defaults to 0.2.
        scale (float, optional): A scalar to adjust the variance of the feature
            map. Defaults to 2**0.5.

    Returns:
        torch.Tensor: Feature map after non-linear activation.
    """

    if not input.is_cuda:
        return bias_leakyrelu_ref(input, bias, negative_slope, scale)

    return FusedBiasLeakyReLUFunction.apply(input, bias.to(input.dtype),
                                            negative_slope, scale)


def bias_leakyrelu_ref(x, bias, negative_slope=0.2, scale=2**0.5):

    if bias is not None:
        assert bias.ndim == 1
        assert bias.shape[0] == x.shape[1]
        x = x + bias.reshape([-1 if i == 1 else 1 for i in range(x.ndim)])

    x = F.leaky_relu(x, negative_slope)
    if scale != 1:
        x = x * scale

    return x
