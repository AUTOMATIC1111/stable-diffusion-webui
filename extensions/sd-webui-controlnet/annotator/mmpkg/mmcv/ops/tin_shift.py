# Copyright (c) OpenMMLab. All rights reserved.
# Code reference from "Temporal Interlacing Network"
# https://github.com/deepcs233/TIN/blob/master/cuda_shift/rtc_wrap.py
# Hao Shao, Shengju Qian, Yu Liu
# shaoh19@mails.tsinghua.edu.cn, sjqian@cse.cuhk.edu.hk, yuliu@ee.cuhk.edu.hk

import torch
import torch.nn as nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['tin_shift_forward', 'tin_shift_backward'])


class TINShiftFunction(Function):

    @staticmethod
    def forward(ctx, input, shift):
        C = input.size(2)
        num_segments = shift.size(1)
        if C // num_segments <= 0 or C % num_segments != 0:
            raise ValueError('C should be a multiple of num_segments, '
                             f'but got C={C} and num_segments={num_segments}.')

        ctx.save_for_backward(shift)

        out = torch.zeros_like(input)
        ext_module.tin_shift_forward(input, shift, out)

        return out

    @staticmethod
    def backward(ctx, grad_output):

        shift = ctx.saved_tensors[0]
        data_grad_input = grad_output.new(*grad_output.size()).zero_()
        shift_grad_input = shift.new(*shift.size()).zero_()
        ext_module.tin_shift_backward(grad_output, shift, data_grad_input)

        return data_grad_input, shift_grad_input


tin_shift = TINShiftFunction.apply


class TINShift(nn.Module):
    """Temporal Interlace Shift.

    Temporal Interlace shift is a differentiable temporal-wise frame shifting
    which is proposed in "Temporal Interlacing Network"

    Please refer to https://arxiv.org/abs/2001.06499 for more details.
    Code is modified from https://github.com/mit-han-lab/temporal-shift-module
    """

    def forward(self, input, shift):
        """Perform temporal interlace shift.

        Args:
            input (Tensor): Feature map with shape [N, num_segments, C, H * W].
            shift (Tensor): Shift tensor with shape [N, num_segments].

        Returns:
            Feature map after temporal interlace shift.
        """
        return tin_shift(input, shift)
