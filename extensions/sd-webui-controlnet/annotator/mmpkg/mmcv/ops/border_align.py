# Copyright (c) OpenMMLab. All rights reserved.
# modified from
# https://github.com/Megvii-BaseDetection/cvpods/blob/master/cvpods/layers/border_align.py

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['border_align_forward', 'border_align_backward'])


class BorderAlignFunction(Function):

    @staticmethod
    def symbolic(g, input, boxes, pool_size):
        return g.op(
            'mmcv::MMCVBorderAlign', input, boxes, pool_size_i=pool_size)

    @staticmethod
    def forward(ctx, input, boxes, pool_size):
        ctx.pool_size = pool_size
        ctx.input_shape = input.size()

        assert boxes.ndim == 3, 'boxes must be with shape [B, H*W, 4]'
        assert boxes.size(2) == 4, \
            'the last dimension of boxes must be (x1, y1, x2, y2)'
        assert input.size(1) % 4 == 0, \
            'the channel for input feature must be divisible by factor 4'

        # [B, C//4, H*W, 4]
        output_shape = (input.size(0), input.size(1) // 4, boxes.size(1), 4)
        output = input.new_zeros(output_shape)
        # `argmax_idx` only used for backward
        argmax_idx = input.new_zeros(output_shape).to(torch.int)

        ext_module.border_align_forward(
            input, boxes, output, argmax_idx, pool_size=ctx.pool_size)

        ctx.save_for_backward(boxes, argmax_idx)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        boxes, argmax_idx = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        # complex head architecture may cause grad_output uncontiguous
        grad_output = grad_output.contiguous()
        ext_module.border_align_backward(
            grad_output,
            boxes,
            argmax_idx,
            grad_input,
            pool_size=ctx.pool_size)
        return grad_input, None, None


border_align = BorderAlignFunction.apply


class BorderAlign(nn.Module):
    r"""Border align pooling layer.

    Applies border_align over the input feature based on predicted bboxes.
    The details were described in the paper
    `BorderDet: Border Feature for Dense Object Detection
    <https://arxiv.org/abs/2007.11056>`_.

    For each border line (e.g. top, left, bottom or right) of each box,
    border_align does the following:
        1. uniformly samples `pool_size`+1 positions on this line, involving \
           the start and end points.
        2. the corresponding features on these points are computed by \
           bilinear interpolation.
        3. max pooling over all the `pool_size`+1 positions are used for \
           computing pooled feature.

    Args:
        pool_size (int): number of positions sampled over the boxes' borders
            (e.g. top, bottom, left, right).

    """

    def __init__(self, pool_size):
        super(BorderAlign, self).__init__()
        self.pool_size = pool_size

    def forward(self, input, boxes):
        """
        Args:
            input: Features with shape [N,4C,H,W]. Channels ranged in [0,C),
                [C,2C), [2C,3C), [3C,4C) represent the top, left, bottom,
                right features respectively.
            boxes: Boxes with shape [N,H*W,4]. Coordinate format (x1,y1,x2,y2).

        Returns:
            Tensor: Pooled features with shape [N,C,H*W,4]. The order is
                (top,left,bottom,right) for the last dimension.
        """
        return border_align(input, boxes, self.pool_size)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(pool_size={self.pool_size})'
        return s
