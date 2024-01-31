# modified from https://github.com/rosinality/stylegan2-pytorch/blob/master/op/upfirdn2d.py  # noqa:E501

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
from torch.autograd import Function
from torch.nn import functional as F

from annotator.mmpkg.mmcv.utils import to_2tuple
from ..utils import ext_loader

upfirdn2d_ext = ext_loader.load_ext('_ext', ['upfirdn2d'])


class UpFirDn2dBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad,
                in_size, out_size):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_ext.upfirdn2d(
            grad_output,
            grad_kernel,
            up_x=down_x,
            up_y=down_y,
            down_x=up_x,
            down_y=up_y,
            pad_x0=g_pad_x0,
            pad_x1=g_pad_x1,
            pad_y0=g_pad_y0,
            pad_y1=g_pad_y1)
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2],
                                     in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2],
                                                ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_ext.upfirdn2d(
            gradgrad_input,
            kernel,
            up_x=ctx.up_x,
            up_y=ctx.up_y,
            down_x=ctx.down_x,
            down_y=ctx.down_y,
            pad_x0=ctx.pad_x0,
            pad_x1=ctx.pad_x1,
            pad_y0=ctx.pad_y0,
            pad_y1=ctx.pad_y1)
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0],
        #                                  ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.in_size[1],
                                         ctx.out_size[0], ctx.out_size[1])

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):

    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_ext.upfirdn2d(
            input,
            kernel,
            up_x=up_x,
            up_y=up_y,
            down_x=down_x,
            down_y=down_y,
            pad_x0=pad_x0,
            pad_x1=pad_x1,
            pad_y0=pad_y0,
            pad_y1=pad_y1)
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """UpFRIDn for 2d features.

    UpFIRDn is short for upsample, apply FIR filter and downsample. More
    details can be found in:
    https://www.mathworks.com/help/signal/ref/upfirdn.html

    Args:
        input (Tensor): Tensor with shape of (n, c, h, w).
        kernel (Tensor): Filter kernel.
        up (int | tuple[int], optional): Upsampling factor. If given a number,
            we will use this factor for the both height and width side.
            Defaults to 1.
        down (int | tuple[int], optional): Downsampling factor. If given a
            number, we will use this factor for the both height and width side.
            Defaults to 1.
        pad (tuple[int], optional): Padding for tensors, (x_pad, y_pad) or
            (x_pad_0, x_pad_1, y_pad_0, y_pad_1). Defaults to (0, 0).

    Returns:
        Tensor: Tensor after UpFIRDn.
    """
    if input.device.type == 'cpu':
        if len(pad) == 2:
            pad = (pad[0], pad[1], pad[0], pad[1])

        up = to_2tuple(up)

        down = to_2tuple(down)

        out = upfirdn2d_native(input, kernel, up[0], up[1], down[0], down[1],
                               pad[0], pad[1], pad[2], pad[3])
    else:
        _up = to_2tuple(up)

        _down = to_2tuple(down)

        if len(pad) == 4:
            _pad = pad
        elif len(pad) == 2:
            _pad = (pad[0], pad[1], pad[0], pad[1])

        out = UpFirDn2d.apply(input, kernel, _up, _down, _pad)

    return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1,
                     pad_y0, pad_y1):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out,
        [0, 0,
         max(pad_x0, 0),
         max(pad_x1, 0),
         max(pad_y0, 0),
         max(pad_y1, 0)])
    out = out[:,
              max(-pad_y0, 0):out.shape[1] - max(-pad_y1, 0),
              max(-pad_x0, 0):out.shape[2] - max(-pad_x1, 0), :, ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
