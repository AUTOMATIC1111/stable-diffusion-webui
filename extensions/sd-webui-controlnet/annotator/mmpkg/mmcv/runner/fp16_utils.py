# Copyright (c) OpenMMLab. All rights reserved.
import functools
import warnings
from collections import abc
from inspect import getfullargspec

import numpy as np
import torch
import torch.nn as nn

from annotator.mmpkg.mmcv.utils import TORCH_VERSION, digit_version
from .dist_utils import allreduce_grads as _allreduce_grads

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.autocast would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    # Note that when PyTorch >= 1.6.0, we still cast tensor types to fp16
    # manually, so the behavior may not be consistent with real amp.
    from torch.cuda.amp import autocast
except ImportError:
    pass


def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, nn.Module):
        return inputs
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


def auto_fp16(apply_to=None, out_fp32=False):
    """Decorator to enable fp16 training automatically.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If inputs arguments are fp32 tensors, they will
    be converted to fp16 automatically. Arguments other than fp32 tensors are
    ignored. If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp32 (bool): Whether to convert the output back to fp32.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp16
        >>>     @auto_fp16()
        >>>     def forward(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp16
        >>>     @auto_fp16(apply_to=('pred', ))
        >>>     def do_something(self, pred, others):
        >>>         pass
    """

    def auto_fp16_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@auto_fp16 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)

            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            # NOTE: default args are not taken into consideration
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.float, torch.half))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = {}
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.float, torch.half)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if (TORCH_VERSION != 'parrots' and
                    digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                with autocast(enabled=True):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp32:
                output = cast_tensor_type(output, torch.half, torch.float)
            return output

        return new_func

    return auto_fp16_wrapper


def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored. If you are using PyTorch >= 1.6,
    torch.cuda.amp is used as the backend, otherwise, original mmcv
    implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp32
        >>>     @force_fp32()
        >>>     def loss(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp32
        >>>     @force_fp32(apply_to=('pred', ))
        >>>     def post_process(self, pred, others):
        >>>         pass
    """

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            if (TORCH_VERSION != 'parrots' and
                    digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                with autocast(enabled=False):
                    output = old_func(*new_args, **new_kwargs)
            else:
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    warnings.warning(
        '"mmcv.runner.fp16_utils.allreduce_grads" is deprecated, and will be '
        'removed in v2.8. Please switch to "mmcv.runner.allreduce_grads')
    _allreduce_grads(params, coalesce=coalesce, bucket_size_mb=bucket_size_mb)


def wrap_fp16_model(model):
    """Wrap the FP32 model to FP16.

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the
    backend, otherwise, original mmcv implementation will be adopted.

    For PyTorch >= 1.6, this function will
    1. Set fp16 flag inside the model to True.

    Otherwise:
    1. Convert FP32 model to FP16.
    2. Remain some necessary layers to be FP32, e.g., normalization layers.
    3. Set `fp16_enabled` flag inside the model to True.

    Args:
        model (nn.Module): Model in FP32.
    """
    if (TORCH_VERSION == 'parrots'
            or digit_version(TORCH_VERSION) < digit_version('1.6.0')):
        # convert model to fp16
        model.half()
        # patch the normalization layers to make it work in fp32 mode
        patch_norm_fp32(model)
    # set `fp16_enabled` flag
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True


def patch_norm_fp32(module):
    """Recursively convert normalization layers from FP16 to FP32.

    Args:
        module (nn.Module): The modules to be converted in FP16.

    Returns:
        nn.Module: The converted module, the normalization layers have been
            converted to FP32.
    """
    if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
        module.float()
        if isinstance(module, nn.GroupNorm) or torch.__version__ < '1.3':
            module.forward = patch_forward_method(module.forward, torch.half,
                                                  torch.float)
    for child in module.children():
        patch_norm_fp32(child)
    return module


def patch_forward_method(func, src_type, dst_type, convert_output=True):
    """Patch the forward method of a module.

    Args:
        func (callable): The original forward method.
        src_type (torch.dtype): Type of input arguments to be converted from.
        dst_type (torch.dtype): Type of input arguments to be converted to.
        convert_output (bool): Whether to convert the output back to src_type.

    Returns:
        callable: The patched forward method.
    """

    def new_forward(*args, **kwargs):
        output = func(*cast_tensor_type(args, src_type, dst_type),
                      **cast_tensor_type(kwargs, src_type, dst_type))
        if convert_output:
            output = cast_tensor_type(output, dst_type, src_type)
        return output

    return new_forward


class LossScaler:
    """Class that manages loss scaling in mixed precision training which
    supports both dynamic or static mode.

    The implementation refers to
    https://github.com/NVIDIA/apex/blob/master/apex/fp16_utils/loss_scaler.py.
    Indirectly, by supplying ``mode='dynamic'`` for dynamic loss scaling.
    It's important to understand how :class:`LossScaler` operates.
    Loss scaling is designed to combat the problem of underflowing
    gradients encountered at long times when training fp16 networks.
    Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.
    If overflowing gradients are encountered, :class:`FP16_Optimizer` then
    skips the update step for this particular iteration/minibatch,
    and :class:`LossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients
    detected,:class:`LossScaler` increases the loss scale once more.
    In this way :class:`LossScaler` attempts to "ride the edge" of always
    using the highest loss scale possible without incurring overflow.

    Args:
        init_scale (float): Initial loss scale value, default: 2**32.
        scale_factor (float): Factor used when adjusting the loss scale.
            Default: 2.
        mode (str): Loss scaling mode. 'dynamic' or 'static'
        scale_window (int): Number of consecutive iterations without an
            overflow to wait before increasing the loss scale. Default: 1000.
    """

    def __init__(self,
                 init_scale=2**32,
                 mode='dynamic',
                 scale_factor=2.,
                 scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        assert mode in ('dynamic',
                        'static'), 'mode can only be dynamic or static'
        self.mode = mode
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    def has_overflow(self, params):
        """Check if params contain overflow."""
        if self.mode != 'dynamic':
            return False
        for p in params:
            if p.grad is not None and LossScaler._has_inf_or_nan(p.grad.data):
                return True
        return False

    def _has_inf_or_nan(x):
        """Check if params contain NaN."""
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if 'value cannot be converted' not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') \
                    or cpu_sum != cpu_sum:
                return True
            return False

    def update_scale(self, overflow):
        """update the current loss scale value when overflow happens."""
        if self.mode != 'dynamic':
            return
        if overflow:
            self.cur_scale = max(self.cur_scale / self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) % \
                    self.scale_window == 0:
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    def state_dict(self):
        """Returns the state of the scaler as a :class:`dict`."""
        return dict(
            cur_scale=self.cur_scale,
            cur_iter=self.cur_iter,
            mode=self.mode,
            last_overflow_iter=self.last_overflow_iter,
            scale_factor=self.scale_factor,
            scale_window=self.scale_window)

    def load_state_dict(self, state_dict):
        """Loads the loss_scaler state dict.

        Args:
           state_dict (dict): scaler state.
        """
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        self.mode = state_dict['mode']
        self.last_overflow_iter = state_dict['last_overflow_iter']
        self.scale_factor = state_dict['scale_factor']
        self.scale_window = state_dict['scale_window']

    @property
    def loss_scale(self):
        return self.cur_scale
