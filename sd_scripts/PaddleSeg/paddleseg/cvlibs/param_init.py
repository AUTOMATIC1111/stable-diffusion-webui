# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn
import math


def uniform_init(param, **kwargs):
    """
    Initialize the `param` with uniform distribution.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 2)
        param_init.uniform_init(linear.bias,  low=-0.5, high=0。5)
        print(linear.bias.numpy())
        # result is [-0.2734719   0.23939109]

    """
    initializer = nn.initializer.Uniform(**kwargs)
    initializer(param, param.block)


def constant_init(param, **kwargs):
    """
    Initialize the `param` with constants.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.constant_init(linear.weight, value=2.0)
        print(linear.weight.numpy())
        # result is [[2. 2. 2. 2.], [2. 2. 2. 2.]]

    """
    initializer = nn.initializer.Constant(**kwargs)
    initializer(param, param.block)


def normal_init(param, **kwargs):
    """
    Initialize the `param` with a Normal distribution.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.normal_init(linear.weight, loc=0.0, scale=1.0)

    """
    initializer = nn.initializer.Normal(**kwargs)
    initializer(param, param.block)


def kaiming_normal_init(param, **kwargs):
    r"""
    Initialize the input tensor with Kaiming Normal initialization.

    This function implements the `param` initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities. In case of Uniform distribution, the range is [-x, x], where
    .. math::
        x = \sqrt{\\frac{6.0}{fan\_in}}
    In case of Normal distribution, the mean is 0 and the standard deviation
    is
    .. math::
        \sqrt{\\frac{2.0}{fan\_in}}

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        # uniform is used to decide whether to use uniform or normal distribution
        param_init.kaiming_normal_init(linear.weight)

    """
    initializer = nn.initializer.KaimingNormal(**kwargs)
    initializer(param, param.block)


def trunc_normal_init(param, **kwargs):
    r"""
    Initialize the input tensor with The Random TruncatedNormal (Gaussian) distribution initializer.

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.trunc_normal_init(linear.weight, mean=0.0, std=0.02)

    """
    initializer = nn.initializer.TruncatedNormal(**kwargs)
    initializer(param, param.block)


def kaiming_uniform(param, **kwargs):
    r"""Implements the Kaiming Uniform initializer
    This class implements the weight initialization from the paper
    `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun. This is a
    robust initialization method that particularly considers the rectifier
    nonlinearities.

    In case of Uniform distribution, the range is [-x, x], where
    .. math::
        x = \sqrt{\\frac{6.0}{fan\_in}}

    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.kaiming_uniform(linear.weight)
    """

    initializer = nn.initializer.KaimingUniform(**kwargs)
    initializer(param, param.block)


def xavier_uniform(param, **kwargs):
    r"""
    This implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.
    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where
    .. math::
        x = \sqrt{\frac{6.0}{fan\_in + fan\_out}}
    Args:
        param (Tensor): Tensor that needs to be initialized.

    Examples:

        from paddleseg.cvlibs import param_init
        import paddle.nn as nn

        linear = nn.Linear(2, 4)
        param_init.xavier_uniform(linear.weight)
    """
    initializer = nn.initializer.XavierUniform(**kwargs)
    initializer(param, param.block)


def multihead_fill(layer, qkv_same_embed_dim=True):
    """
    The default initialization of multi-head attention.

    Example:
        from paddleseg.cvlibs import param_init
        import paddle.nn as nn
        
        self_attn = nn.MultiHeadAttention(
            128, 8, dropout=False)
        param_init.multihead_fill(self_attn, True)
    """

    def _init_param_as_combined_linear_weight(p):
        bound = math.sqrt(6 / (3 * p.shape[0] + p.shape[1]))
        nn.initializer.Uniform(low=-bound, high=bound)(p)

    if qkv_same_embed_dim:
        _init_param_as_combined_linear_weight(layer.q_proj.weight)
        _init_param_as_combined_linear_weight(layer.k_proj.weight)
        _init_param_as_combined_linear_weight(layer.v_proj.weight)
        xavier_uniform(layer.out_proj.weight)
    else:
        for p in layer.parameters():
            if p.dim() > 1:
                xavier_uniform(p)


def th_linear_fill(layer):
    """
    The default way of linear initialization.
    
    Example:
        from paddleseg.cvlibs import param_init
        import paddle.nn as nn
        
        linear = nn.Linear(128, 128)
        param_init.linear_fill(linear)
    """
    nn.initializer.KaimingUniform(
        negative_slope=math.sqrt(5), nonlinearity='leaky_relu')(layer.weight)

    if getattr(layer, 'bias', None) is not None:
        fan_in = layer.weight.shape[0]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.initializer.Uniform(low=-bound, high=bound)(layer.bias)
