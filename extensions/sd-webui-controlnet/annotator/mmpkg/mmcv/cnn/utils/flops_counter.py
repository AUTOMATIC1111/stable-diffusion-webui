# Modified from flops-counter.pytorch by Vladislav Sovrasov
# original repo: https://github.com/sovrasov/flops-counter.pytorch

# MIT License

# Copyright (c) 2018 Vladislav Sovrasov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
from functools import partial

import numpy as np
import torch
import torch.nn as nn

import annotator.mmpkg.mmcv as mmcv


def get_model_complexity_info(model,
                              input_shape,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None,
                              flush=False,
                              ost=sys.stdout):
    """Get complexity information of a model.

    This method can calculate FLOPs and parameter counts of a model with
    corresponding input shape. It can also print complexity information for
    each layer in a model.

    Supported layers are listed as below:
        - Convolutions: ``nn.Conv1d``, ``nn.Conv2d``, ``nn.Conv3d``.
        - Activations: ``nn.ReLU``, ``nn.PReLU``, ``nn.ELU``, ``nn.LeakyReLU``,
            ``nn.ReLU6``.
        - Poolings: ``nn.MaxPool1d``, ``nn.MaxPool2d``, ``nn.MaxPool3d``,
            ``nn.AvgPool1d``, ``nn.AvgPool2d``, ``nn.AvgPool3d``,
            ``nn.AdaptiveMaxPool1d``, ``nn.AdaptiveMaxPool2d``,
            ``nn.AdaptiveMaxPool3d``, ``nn.AdaptiveAvgPool1d``,
            ``nn.AdaptiveAvgPool2d``, ``nn.AdaptiveAvgPool3d``.
        - BatchNorms: ``nn.BatchNorm1d``, ``nn.BatchNorm2d``,
            ``nn.BatchNorm3d``, ``nn.GroupNorm``, ``nn.InstanceNorm1d``,
            ``InstanceNorm2d``, ``InstanceNorm3d``, ``nn.LayerNorm``.
        - Linear: ``nn.Linear``.
        - Deconvolution: ``nn.ConvTranspose2d``.
        - Upsample: ``nn.Upsample``.

    Args:
        model (nn.Module): The model for complexity calculation.
        input_shape (tuple): Input shape used for calculation.
        print_per_layer_stat (bool): Whether to print complexity information
            for each layer in a model. Default: True.
        as_strings (bool): Output FLOPs and params counts in a string form.
            Default: True.
        input_constructor (None | callable): If specified, it takes a callable
            method that generates input. otherwise, it will generate a random
            tensor with input shape to calculate FLOPs. Default: None.
        flush (bool): same as that in :func:`print`. Default: False.
        ost (stream): same as ``file`` param in :func:`print`.
            Default: sys.stdout.

    Returns:
        tuple[float | str]: If ``as_strings`` is set to True, it will return
            FLOPs and parameter counts in a string format. otherwise, it will
            return those in a float number format.
    """
    assert type(input_shape) is tuple
    assert len(input_shape) >= 1
    assert isinstance(model, nn.Module)
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()
    if input_constructor:
        input = input_constructor(input_shape)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty(
                (1, *input_shape),
                dtype=next(flops_model.parameters()).dtype,
                device=next(flops_model.parameters()).device)
        except StopIteration:
            # Avoid StopIteration for models which have no parameters,
            # like `nn.Relu()`, `nn.AvgPool2d`, etc.
            batch = torch.ones(()).new_empty((1, *input_shape))

        _ = flops_model(batch)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(
            flops_model, flops_count, params_count, ost=ost, flush=flush)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GFLOPs', precision=2):
    """Convert FLOPs number into a string.

    Note that Here we take a multiply-add counts as one FLOP.

    Args:
        flops (float): FLOPs number to be converted.
        units (str | None): Converted FLOPs units. Options are None, 'GFLOPs',
            'MFLOPs', 'KFLOPs', 'FLOPs'. If set to None, it will automatically
            choose the most suitable unit for FLOPs. Default: 'GFLOPs'.
        precision (int): Digit number after the decimal point. Default: 2.

    Returns:
        str: The converted FLOPs number with units.

    Examples:
        >>> flops_to_string(1e9)
        '1.0 GFLOPs'
        >>> flops_to_string(2e5, 'MFLOPs')
        '0.2 MFLOPs'
        >>> flops_to_string(3e-9, None)
        '3e-09 FLOPs'
    """
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GFLOPs'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MFLOPs'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KFLOPs'
        else:
            return str(flops) + ' FLOPs'
    else:
        if units == 'GFLOPs':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MFLOPs':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KFLOPs':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' FLOPs'


def params_to_string(num_params, units=None, precision=2):
    """Convert parameter number into a string.

    Args:
        num_params (float): Parameter number to be converted.
        units (str | None): Converted FLOPs units. Options are None, 'M',
            'K' and ''. If set to None, it will automatically choose the most
            suitable unit for Parameter number. Default: None.
        precision (int): Digit number after the decimal point. Default: 2.

    Returns:
        str: The converted parameter number with units.

    Examples:
        >>> params_to_string(1e9)
        '1000.0 M'
        >>> params_to_string(2e5)
        '200.0 k'
        >>> params_to_string(3e-9)
        '3e-09'
    """
    if units is None:
        if num_params // 10**6 > 0:
            return str(round(num_params / 10**6, precision)) + ' M'
        elif num_params // 10**3:
            return str(round(num_params / 10**3, precision)) + ' k'
        else:
            return str(num_params)
    else:
        if units == 'M':
            return str(round(num_params / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(num_params / 10.**3, precision)) + ' ' + units
        else:
            return str(num_params)


def print_model_with_flops(model,
                           total_flops,
                           total_params,
                           units='GFLOPs',
                           precision=3,
                           ost=sys.stdout,
                           flush=False):
    """Print a model with FLOPs for each layer.

    Args:
        model (nn.Module): The model to be printed.
        total_flops (float): Total FLOPs of the model.
        total_params (float): Total parameter counts of the model.
        units (str | None): Converted FLOPs units. Default: 'GFLOPs'.
        precision (int): Digit number after the decimal point. Default: 3.
        ost (stream): same as `file` param in :func:`print`.
            Default: sys.stdout.
        flush (bool): same as that in :func:`print`. Default: False.

    Example:
        >>> class ExampleModel(nn.Module):

        >>> def __init__(self):
        >>>     super().__init__()
        >>>     self.conv1 = nn.Conv2d(3, 8, 3)
        >>>     self.conv2 = nn.Conv2d(8, 256, 3)
        >>>     self.conv3 = nn.Conv2d(256, 8, 3)
        >>>     self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        >>>     self.flatten = nn.Flatten()
        >>>     self.fc = nn.Linear(8, 1)

        >>> def forward(self, x):
        >>>     x = self.conv1(x)
        >>>     x = self.conv2(x)
        >>>     x = self.conv3(x)
        >>>     x = self.avg_pool(x)
        >>>     x = self.flatten(x)
        >>>     x = self.fc(x)
        >>>     return x

        >>> model = ExampleModel()
        >>> x = (3, 16, 16)
        to print the complexity information state for each layer, you can use
        >>> get_model_complexity_info(model, x)
        or directly use
        >>> print_model_with_flops(model, 4579784.0, 37361)
        ExampleModel(
          0.037 M, 100.000% Params, 0.005 GFLOPs, 100.000% FLOPs,
          (conv1): Conv2d(0.0 M, 0.600% Params, 0.0 GFLOPs, 0.959% FLOPs, 3, 8, kernel_size=(3, 3), stride=(1, 1))  # noqa: E501
          (conv2): Conv2d(0.019 M, 50.020% Params, 0.003 GFLOPs, 58.760% FLOPs, 8, 256, kernel_size=(3, 3), stride=(1, 1))
          (conv3): Conv2d(0.018 M, 49.356% Params, 0.002 GFLOPs, 40.264% FLOPs, 256, 8, kernel_size=(3, 3), stride=(1, 1))
          (avg_pool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.017% FLOPs, output_size=(1, 1))
          (flatten): Flatten(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          (fc): Linear(0.0 M, 0.024% Params, 0.0 GFLOPs, 0.000% FLOPs, in_features=8, out_features=1, bias=True)
        )
    """

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_num_params = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([
            params_to_string(
                accumulated_num_params, units='M', precision=precision),
            '{:.3%} Params'.format(accumulated_num_params / total_params),
            flops_to_string(
                accumulated_flops_cost, units=units, precision=precision),
            '{:.3%} FLOPs'.format(accumulated_flops_cost / total_flops),
            self.original_extra_repr()
        ])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model, file=ost, flush=flush)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    """Calculate parameter number of a model.

    Args:
        model (nn.module): The model for parameter number calculation.

    Returns:
        float: Parameter number of the model.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(
        net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(
        net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(
        net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(  # noqa: E501
        net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module


def compute_average_flops_cost(self):
    """Compute average FLOPs cost.

    A method to compute average FLOPs cost, which will be available after
    `add_flops_counting_methods()` is called on a desired net object.

    Returns:
        float: Current mean flops consumption per image.
    """
    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__
    params_sum = get_model_parameters_number(self)
    return flops_sum / batches_count, params_sum


def start_flops_count(self):
    """Activate the computation of mean flops consumption per image.

    A method to activate the computation of mean flops consumption per image.
    which will be available after ``add_flops_counting_methods()`` is called on
    a desired net object. It should be called before running the network.
    """
    add_batch_counter_hook_function(self)

    def add_flops_counter_hook_function(module):
        if is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return

            else:
                handle = module.register_forward_hook(
                    get_modules_mapping()[type(module)])

            module.__flops_handle__ = handle

    self.apply(partial(add_flops_counter_hook_function))


def stop_flops_count(self):
    """Stop computing the mean flops consumption per image.

    A method to stop computing the mean flops consumption per image, which will
    be available after ``add_flops_counting_methods()`` is called on a desired
    net object. It can be called to pause the computation whenever.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """Reset statistics computed so far.

    A method to Reset computed statistics, which will be available after
    `add_flops_counting_methods()` is called on a desired net object.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    output_last_dim = output.shape[
        -1]  # pytorch checks dimensions, so here we don't care much
    module.__flops__ += int(np.prod(input.shape) * output_last_dim)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def norm_flops_counter_hook(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    if (getattr(module, 'affine', False)
            or getattr(module, 'elementwise_affine', False)):
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def deconv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    input_height, input_width = input.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = (
        kernel_height * kernel_width * in_channels * filters_per_channel)

    active_elements_count = batch_size * input_height * input_width
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        output_height, output_width = output.shape[2:]
        bias_flops = out_channels * batch_size * output_height * output_height
    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(
        np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module, '
              'assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__'):
            print('Warning: variables __flops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptflops can affect your code!')
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in get_modules_mapping():
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


def get_modules_mapping():
    return {
        # convolutions
        nn.Conv1d: conv_flops_counter_hook,
        nn.Conv2d: conv_flops_counter_hook,
        mmcv.cnn.bricks.Conv2d: conv_flops_counter_hook,
        nn.Conv3d: conv_flops_counter_hook,
        mmcv.cnn.bricks.Conv3d: conv_flops_counter_hook,
        # activations
        nn.ReLU: relu_flops_counter_hook,
        nn.PReLU: relu_flops_counter_hook,
        nn.ELU: relu_flops_counter_hook,
        nn.LeakyReLU: relu_flops_counter_hook,
        nn.ReLU6: relu_flops_counter_hook,
        # poolings
        nn.MaxPool1d: pool_flops_counter_hook,
        nn.AvgPool1d: pool_flops_counter_hook,
        nn.AvgPool2d: pool_flops_counter_hook,
        nn.MaxPool2d: pool_flops_counter_hook,
        mmcv.cnn.bricks.MaxPool2d: pool_flops_counter_hook,
        nn.MaxPool3d: pool_flops_counter_hook,
        mmcv.cnn.bricks.MaxPool3d: pool_flops_counter_hook,
        nn.AvgPool3d: pool_flops_counter_hook,
        nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
        nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
        nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
        nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
        nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
        nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
        # normalizations
        nn.BatchNorm1d: norm_flops_counter_hook,
        nn.BatchNorm2d: norm_flops_counter_hook,
        nn.BatchNorm3d: norm_flops_counter_hook,
        nn.GroupNorm: norm_flops_counter_hook,
        nn.InstanceNorm1d: norm_flops_counter_hook,
        nn.InstanceNorm2d: norm_flops_counter_hook,
        nn.InstanceNorm3d: norm_flops_counter_hook,
        nn.LayerNorm: norm_flops_counter_hook,
        # FC
        nn.Linear: linear_flops_counter_hook,
        mmcv.cnn.bricks.Linear: linear_flops_counter_hook,
        # Upscale
        nn.Upsample: upsample_flops_counter_hook,
        # Deconvolution
        nn.ConvTranspose2d: deconv_flops_counter_hook,
        mmcv.cnn.bricks.ConvTranspose2d: deconv_flops_counter_hook,
    }
