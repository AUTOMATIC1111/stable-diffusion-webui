import logging

import annotator.mmpkg.mmcv as mmcv
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import ConvModule, constant_init, kaiming_init
from annotator.mmpkg.mmcv.cnn.bricks import Conv2dAdaptivePadding
from annotator.mmpkg.mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import InvertedResidualV3 as InvertedResidual


@BACKBONES.register_module()
class MobileNetV3(nn.Module):
    """MobileNetV3 backbone.

    This backbone is the improved implementation of `Searching for MobileNetV3
    <https://ieeexplore.ieee.org/document/9008835>`_.

    Args:
        arch (str): Architecture of mobilnetv3, from {'small', 'large'}.
            Default: 'small'.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        out_indices (tuple[int]): Output from which layer.
            Default: (0, 1, 12).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed.
            Default: False.
    """
    # Parameters to build each block:
    #     [kernel size, mid channels, out channels, with_se, act type, stride]
    arch_settings = {
        'small': [[3, 16, 16, True, 'ReLU', 2],  # block0 layer1 os=4
                  [3, 72, 24, False, 'ReLU', 2],  # block1 layer2 os=8
                  [3, 88, 24, False, 'ReLU', 1],
                  [5, 96, 40, True, 'HSwish', 2],  # block2 layer4 os=16
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 120, 48, True, 'HSwish', 1],  # block3 layer7 os=16
                  [5, 144, 48, True, 'HSwish', 1],
                  [5, 288, 96, True, 'HSwish', 2],  # block4 layer9 os=32
                  [5, 576, 96, True, 'HSwish', 1],
                  [5, 576, 96, True, 'HSwish', 1]],
        'large': [[3, 16, 16, False, 'ReLU', 1],  # block0 layer1 os=2
                  [3, 64, 24, False, 'ReLU', 2],  # block1 layer2 os=4
                  [3, 72, 24, False, 'ReLU', 1],
                  [5, 72, 40, True, 'ReLU', 2],  # block2 layer4 os=8
                  [5, 120, 40, True, 'ReLU', 1],
                  [5, 120, 40, True, 'ReLU', 1],
                  [3, 240, 80, False, 'HSwish', 2],  # block3 layer7 os=16
                  [3, 200, 80, False, 'HSwish', 1],
                  [3, 184, 80, False, 'HSwish', 1],
                  [3, 184, 80, False, 'HSwish', 1],
                  [3, 480, 112, True, 'HSwish', 1],  # block4 layer11 os=16
                  [3, 672, 112, True, 'HSwish', 1],
                  [5, 672, 160, True, 'HSwish', 2],  # block5 layer13 os=32
                  [5, 960, 160, True, 'HSwish', 1],
                  [5, 960, 160, True, 'HSwish', 1]]
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 out_indices=(0, 1, 12),
                 frozen_stages=-1,
                 reduction_factor=1,
                 norm_eval=False,
                 with_cp=False):
        super(MobileNetV3, self).__init__()
        assert arch in self.arch_settings
        assert isinstance(reduction_factor, int) and reduction_factor > 0
        assert mmcv.is_tuple_of(out_indices, int)
        for index in out_indices:
            if index not in range(0, len(self.arch_settings[arch]) + 2):
                raise ValueError(
                    'the item in out_indices must in '
                    f'range(0, {len(self.arch_settings[arch])+2}). '
                    f'But received {index}')

        if frozen_stages not in range(-1, len(self.arch_settings[arch]) + 2):
            raise ValueError('frozen_stages must be in range(-1, '
                             f'{len(self.arch_settings[arch])+2}). '
                             f'But received {frozen_stages}')
        self.arch = arch
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.reduction_factor = reduction_factor
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.layers = self._make_layer()

    def _make_layer(self):
        layers = []

        # build the first layer (layer0)
        in_channels = 16
        layer = ConvModule(
            in_channels=3,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=dict(type='Conv2dAdaptivePadding'),
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='HSwish'))
        self.add_module('layer0', layer)
        layers.append('layer0')

        layer_setting = self.arch_settings[self.arch]
        for i, params in enumerate(layer_setting):
            (kernel_size, mid_channels, out_channels, with_se, act,
             stride) = params

            if self.arch == 'large' and i >= 12 or self.arch == 'small' and \
                    i >= 8:
                mid_channels = mid_channels // self.reduction_factor
                out_channels = out_channels // self.reduction_factor

            if with_se:
                se_cfg = dict(
                    channels=mid_channels,
                    ratio=4,
                    act_cfg=(dict(type='ReLU'),
                             dict(type='HSigmoid', bias=3.0, divisor=6.0)))
            else:
                se_cfg = None

            layer = InvertedResidual(
                in_channels=in_channels,
                out_channels=out_channels,
                mid_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                se_cfg=se_cfg,
                with_expand_conv=(in_channels != mid_channels),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type=act),
                with_cp=self.with_cp)
            in_channels = out_channels
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, layer)
            layers.append(layer_name)

        # build the last layer
        # block5 layer12 os=32 for small model
        # block6 layer16 os=32 for large model
        layer = ConvModule(
            in_channels=in_channels,
            out_channels=576 if self.arch == 'small' else 960,
            kernel_size=1,
            stride=1,
            dilation=4,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='HSwish'))
        layer_name = 'layer{}'.format(len(layer_setting) + 1)
        self.add_module(layer_name, layer)
        layers.append(layer_name)

        # next, convert backbone MobileNetV3 to a semantic segmentation version
        if self.arch == 'small':
            self.layer4.depthwise_conv.conv.stride = (1, 1)
            self.layer9.depthwise_conv.conv.stride = (1, 1)
            for i in range(4, len(layers)):
                layer = getattr(self, layers[i])
                if isinstance(layer, InvertedResidual):
                    modified_module = layer.depthwise_conv.conv
                else:
                    modified_module = layer.conv

                if i < 9:
                    modified_module.dilation = (2, 2)
                    pad = 2
                else:
                    modified_module.dilation = (4, 4)
                    pad = 4

                if not isinstance(modified_module, Conv2dAdaptivePadding):
                    # Adjust padding
                    pad *= (modified_module.kernel_size[0] - 1) // 2
                    modified_module.padding = (pad, pad)
        else:
            self.layer7.depthwise_conv.conv.stride = (1, 1)
            self.layer13.depthwise_conv.conv.stride = (1, 1)
            for i in range(7, len(layers)):
                layer = getattr(self, layers[i])
                if isinstance(layer, InvertedResidual):
                    modified_module = layer.depthwise_conv.conv
                else:
                    modified_module = layer.conv

                if i < 13:
                    modified_module.dilation = (2, 2)
                    pad = 2
                else:
                    modified_module.dilation = (4, 4)
                    pad = 4

                if not isinstance(modified_module, Conv2dAdaptivePadding):
                    # Adjust padding
                    pad *= (modified_module.kernel_size[0] - 1) // 2
                    modified_module.padding = (pad, pad)

        return layers

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV3, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
