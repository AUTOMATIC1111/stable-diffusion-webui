# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['correlation_forward', 'correlation_backward'])


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx,
                input1,
                input2,
                kernel_size=1,
                max_displacement=1,
                stride=1,
                padding=1,
                dilation=1,
                dilation_patch=1):

        ctx.save_for_backward(input1, input2)

        kH, kW = ctx.kernel_size = _pair(kernel_size)
        patch_size = max_displacement * 2 + 1
        ctx.patch_size = patch_size
        dH, dW = ctx.stride = _pair(stride)
        padH, padW = ctx.padding = _pair(padding)
        dilationH, dilationW = ctx.dilation = _pair(dilation)
        dilation_patchH, dilation_patchW = ctx.dilation_patch = _pair(
            dilation_patch)

        output_size = CorrelationFunction._output_size(ctx, input1)

        output = input1.new_zeros(output_size)

        ext_module.correlation_forward(
            input1,
            input2,
            output,
            kH=kH,
            kW=kW,
            patchH=patch_size,
            patchW=patch_size,
            padH=padH,
            padW=padW,
            dilationH=dilationH,
            dilationW=dilationW,
            dilation_patchH=dilation_patchH,
            dilation_patchW=dilation_patchW,
            dH=dH,
            dW=dW)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        kH, kW = ctx.kernel_size
        patch_size = ctx.patch_size
        padH, padW = ctx.padding
        dilationH, dilationW = ctx.dilation
        dilation_patchH, dilation_patchW = ctx.dilation_patch
        dH, dW = ctx.stride
        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)

        ext_module.correlation_backward(
            grad_output,
            input1,
            input2,
            grad_input1,
            grad_input2,
            kH=kH,
            kW=kW,
            patchH=patch_size,
            patchW=patch_size,
            padH=padH,
            padW=padW,
            dilationH=dilationH,
            dilationW=dilationW,
            dilation_patchH=dilation_patchH,
            dilation_patchW=dilation_patchW,
            dH=dH,
            dW=dW)
        return grad_input1, grad_input2, None, None, None, None, None, None

    @staticmethod
    def _output_size(ctx, input1):
        iH, iW = input1.size(2), input1.size(3)
        batch_size = input1.size(0)
        kH, kW = ctx.kernel_size
        patch_size = ctx.patch_size
        dH, dW = ctx.stride
        padH, padW = ctx.padding
        dilationH, dilationW = ctx.dilation
        dilatedKH = (kH - 1) * dilationH + 1
        dilatedKW = (kW - 1) * dilationW + 1

        oH = int((iH + 2 * padH - dilatedKH) / dH + 1)
        oW = int((iW + 2 * padW - dilatedKW) / dW + 1)

        output_size = (batch_size, patch_size, patch_size, oH, oW)
        return output_size


class Correlation(nn.Module):
    r"""Correlation operator

    This correlation operator works for optical flow correlation computation.

    There are two batched tensors with shape :math:`(N, C, H, W)`,
    and the correlation output's shape is :math:`(N, max\_displacement \times
    2 + 1, max\_displacement * 2 + 1, H_{out}, W_{out})`

    where

    .. math::
        H_{out} = \left\lfloor\frac{H_{in}  + 2 \times padding -
            dilation \times (kernel\_size - 1) - 1}
            {stride} + 1\right\rfloor

    .. math::
        W_{out} = \left\lfloor\frac{W_{in}  + 2 \times padding - dilation
            \times (kernel\_size - 1) - 1}
            {stride} + 1\right\rfloor

    the correlation item :math:`(N_i, dy, dx)` is formed by taking the sliding
    window convolution between input1 and shifted input2,

    .. math::
        Corr(N_i, dx, dy) =
        \sum_{c=0}^{C-1}
        input1(N_i, c) \star
        \mathcal{S}(input2(N_i, c), dy, dx)

    where :math:`\star` is the valid 2d sliding window convolution operator,
    and :math:`\mathcal{S}` means shifting the input features (auto-complete
    zero marginal), and :math:`dx, dy` are shifting distance, :math:`dx, dy \in
    [-max\_displacement \times dilation\_patch, max\_displacement \times
    dilation\_patch]`.

    Args:
        kernel_size (int): The size of sliding window i.e. local neighborhood
            representing the center points and involved in correlation
            computation. Defaults to 1.
        max_displacement (int): The radius for computing correlation volume,
            but the actual working space can be dilated by dilation_patch.
            Defaults to 1.
        stride (int): The stride of the sliding blocks in the input spatial
            dimensions. Defaults to 1.
        padding (int): Zero padding added to all four sides of the input1.
            Defaults to 0.
        dilation (int): The spacing of local neighborhood that will involved
            in correlation. Defaults to 1.
        dilation_patch (int): The spacing between position need to compute
            correlation.  Defaults to 1.
    """

    def __init__(self,
                 kernel_size: int = 1,
                 max_displacement: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 dilation_patch: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return CorrelationFunction.apply(input1, input2, self.kernel_size,
                                         self.max_displacement, self.stride,
                                         self.padding, self.dilation,
                                         self.dilation_patch)

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f'(kernel_size={self.kernel_size}, '
        s += f'max_displacement={self.max_displacement}, '
        s += f'stride={self.stride}, '
        s += f'padding={self.padding}, '
        s += f'dilation={self.dilation}, '
        s += f'dilation_patch={self.dilation_patch})'
        return s
