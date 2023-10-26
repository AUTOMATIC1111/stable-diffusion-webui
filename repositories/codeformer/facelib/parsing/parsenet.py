"""Modified from https://github.com/chaofengc/PSFRGAN
"""
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


class NormLayer(nn.Module):
    """Normalization Layers.

    Args:
        channels: input channels, for batch norm and instance norm.
        input_size: input shape without batch size, for layer norm.
    """

    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        self.norm_type = norm_type
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x * 1.0
        else:
            assert 1 == 0, f'Norm type {norm_type} not support.'

    def forward(self, x, ref=None):
        if self.norm_type == 'spade':
            return self.norm(x, ref)
        else:
            return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.

    Args:
        relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """

    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x * 1.0
        else:
            assert 1 == 0, f'Relu type {relu_type} not support.'

    def forward(self, x):
        return self.func(x)


class ConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 scale='none',
                 norm_type='none',
                 relu_type='none',
                 use_pad=True,
                 bias=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad
        self.norm_type = norm_type
        if norm_type in ['bn']:
            bias = False

        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(int(np.ceil((kernel_size - 1.) / 2)))
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none'):
        super(ResidualBlock, self).__init__()

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        scale_config_dict = {'down': ['none', 'down'], 'up': ['up', 'none'], 'none': ['none', 'none']}
        scale_conf = scale_config_dict[scale]

        self.conv1 = ConvLayer(c_in, c_out, 3, scale_conf[0], norm_type=norm_type, relu_type=relu_type)
        self.conv2 = ConvLayer(c_out, c_out, 3, scale_conf[1], norm_type=norm_type, relu_type='none')

    def forward(self, x):
        identity = self.shortcut_func(x)

        res = self.conv1(x)
        res = self.conv2(res)
        return identity + res


class ParseNet(nn.Module):

    def __init__(self,
                 in_size=128,
                 out_size=128,
                 min_feat_size=32,
                 base_ch=64,
                 parsing_ch=19,
                 res_depth=10,
                 relu_type='LeakyReLU',
                 norm_type='bn',
                 ch_range=[32, 256]):
        super().__init__()
        self.res_depth = res_depth
        act_args = {'norm_type': norm_type, 'relu_type': relu_type}
        min_ch, max_ch = ch_range

        ch_clip = lambda x: max(min_ch, min(x, max_ch))  # noqa: E731
        min_feat_size = min(in_size, min_feat_size)

        down_steps = int(np.log2(in_size // min_feat_size))
        up_steps = int(np.log2(out_size // min_feat_size))

        # =============== define encoder-body-decoder ====================
        self.encoder = []
        self.encoder.append(ConvLayer(3, base_ch, 3, 1))
        head_ch = base_ch
        for i in range(down_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch * 2)
            self.encoder.append(ResidualBlock(cin, cout, scale='down', **act_args))
            head_ch = head_ch * 2

        self.body = []
        for i in range(res_depth):
            self.body.append(ResidualBlock(ch_clip(head_ch), ch_clip(head_ch), **act_args))

        self.decoder = []
        for i in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2)
            self.decoder.append(ResidualBlock(cin, cout, scale='up', **act_args))
            head_ch = head_ch // 2

        self.encoder = nn.Sequential(*self.encoder)
        self.body = nn.Sequential(*self.body)
        self.decoder = nn.Sequential(*self.decoder)
        self.out_img_conv = ConvLayer(ch_clip(head_ch), 3)
        self.out_mask_conv = ConvLayer(ch_clip(head_ch), parsing_ch)

    def forward(self, x):
        feat = self.encoder(x)
        x = feat + self.body(feat)
        x = self.decoder(x)
        out_img = self.out_img_conv(x)
        out_mask = self.out_mask_conv(x)
        return out_mask, out_img
