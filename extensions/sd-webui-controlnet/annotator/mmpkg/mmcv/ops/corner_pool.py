# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'top_pool_forward', 'top_pool_backward', 'bottom_pool_forward',
    'bottom_pool_backward', 'left_pool_forward', 'left_pool_backward',
    'right_pool_forward', 'right_pool_backward'
])

_mode_dict = {'top': 0, 'bottom': 1, 'left': 2, 'right': 3}


class TopPoolFunction(Function):

    @staticmethod
    def symbolic(g, input):
        output = g.op(
            'mmcv::MMCVCornerPool', input, mode_i=int(_mode_dict['top']))
        return output

    @staticmethod
    def forward(ctx, input):
        output = ext_module.top_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = ext_module.top_pool_backward(input, grad_output)
        return output


class BottomPoolFunction(Function):

    @staticmethod
    def symbolic(g, input):
        output = g.op(
            'mmcv::MMCVCornerPool', input, mode_i=int(_mode_dict['bottom']))
        return output

    @staticmethod
    def forward(ctx, input):
        output = ext_module.bottom_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = ext_module.bottom_pool_backward(input, grad_output)
        return output


class LeftPoolFunction(Function):

    @staticmethod
    def symbolic(g, input):
        output = g.op(
            'mmcv::MMCVCornerPool', input, mode_i=int(_mode_dict['left']))
        return output

    @staticmethod
    def forward(ctx, input):
        output = ext_module.left_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = ext_module.left_pool_backward(input, grad_output)
        return output


class RightPoolFunction(Function):

    @staticmethod
    def symbolic(g, input):
        output = g.op(
            'mmcv::MMCVCornerPool', input, mode_i=int(_mode_dict['right']))
        return output

    @staticmethod
    def forward(ctx, input):
        output = ext_module.right_pool_forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        output = ext_module.right_pool_backward(input, grad_output)
        return output


class CornerPool(nn.Module):
    """Corner Pooling.

    Corner Pooling is a new type of pooling layer that helps a
    convolutional network better localize corners of bounding boxes.

    Please refer to https://arxiv.org/abs/1808.01244 for more details.
    Code is modified from https://github.com/princeton-vl/CornerNet-Lite.

    Args:
        mode(str): Pooling orientation for the pooling layer

            - 'bottom': Bottom Pooling
            - 'left': Left Pooling
            - 'right': Right Pooling
            - 'top': Top Pooling

    Returns:
        Feature map after pooling.
    """

    pool_functions = {
        'bottom': BottomPoolFunction,
        'left': LeftPoolFunction,
        'right': RightPoolFunction,
        'top': TopPoolFunction,
    }

    cummax_dim_flip = {
        'bottom': (2, False),
        'left': (3, True),
        'right': (3, False),
        'top': (2, True),
    }

    def __init__(self, mode):
        super(CornerPool, self).__init__()
        assert mode in self.pool_functions
        self.mode = mode
        self.corner_pool = self.pool_functions[mode]

    def forward(self, x):
        if torch.__version__ != 'parrots' and torch.__version__ >= '1.5.0':
            if torch.onnx.is_in_onnx_export():
                assert torch.__version__ >= '1.7.0', \
                    'When `cummax` serves as an intermediate component whose '\
                    'outputs is used as inputs for another modules, it\'s '\
                    'expected that pytorch version must be >= 1.7.0, '\
                    'otherwise Error appears like: `RuntimeError: tuple '\
                    'appears in op that does not forward tuples, unsupported '\
                    'kind: prim::PythonOp`.'

            dim, flip = self.cummax_dim_flip[self.mode]
            if flip:
                x = x.flip(dim)
            pool_tensor, _ = torch.cummax(x, dim=dim)
            if flip:
                pool_tensor = pool_tensor.flip(dim)
            return pool_tensor
        else:
            return self.corner_pool.apply(x)
