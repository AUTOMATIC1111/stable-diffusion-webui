import torch
import torch.nn as nn
from annotator.mmpkg.mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, constant_init,
                      kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm

from annotator.mmpkg.mmseg.models.decode_heads.psp_head import PPM
from annotator.mmpkg.mmseg.ops import resize
from ..builder import BACKBONES
from ..utils.inverted_residual import InvertedResidual


class LearningToDownsample(nn.Module):
    """Learning to downsample module.

    Args:
        in_channels (int): Number of input channels.
        dw_channels (tuple[int]): Number of output channels of the first and
            the second depthwise conv (dwconv) layers.
        out_channels (int): Number of output channels of the whole
            'learning to downsample' module.
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 dw_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(LearningToDownsample, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        dw_channels1 = dw_channels[0]
        dw_channels2 = dw_channels[1]

        self.conv = ConvModule(
            in_channels,
            dw_channels1,
            3,
            stride=2,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.dsconv1 = DepthwiseSeparableConvModule(
            dw_channels1,
            dw_channels2,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg)
        self.dsconv2 = DepthwiseSeparableConvModule(
            dw_channels2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module.

    Args:
        in_channels (int): Number of input channels of the GFE module.
            Default: 64
        block_channels (tuple[int]): Tuple of ints. Each int specifies the
            number of output channels of each Inverted Residual module.
            Default: (64, 96, 128)
        out_channels(int): Number of output channels of the GFE module.
            Default: 128
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
            Default: 6
        num_blocks (tuple[int]): Tuple of ints. Each int specifies the
            number of times each Inverted Residual module is repeated.
            The repeated Inverted Residual modules are called a 'group'.
            Default: (3, 3, 3)
        strides (tuple[int]): Tuple of ints. Each int specifies
            the downsampling factor of each 'group'.
            Default: (2, 2, 1)
        pool_scales (tuple[int]): Tuple of ints. Each int specifies
            the parameter required in 'global average pooling' within PPM.
            Default: (1, 2, 3, 6)
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False
    """

    def __init__(self,
                 in_channels=64,
                 block_channels=(64, 96, 128),
                 out_channels=128,
                 expand_ratio=6,
                 num_blocks=(3, 3, 3),
                 strides=(2, 2, 1),
                 pool_scales=(1, 2, 3, 6),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super(GlobalFeatureExtractor, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        assert len(block_channels) == len(num_blocks) == 3
        self.bottleneck1 = self._make_layer(in_channels, block_channels[0],
                                            num_blocks[0], strides[0],
                                            expand_ratio)
        self.bottleneck2 = self._make_layer(block_channels[0],
                                            block_channels[1], num_blocks[1],
                                            strides[1], expand_ratio)
        self.bottleneck3 = self._make_layer(block_channels[1],
                                            block_channels[2], num_blocks[2],
                                            strides[2], expand_ratio)
        self.ppm = PPM(
            pool_scales,
            block_channels[2],
            block_channels[2] // 4,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=align_corners)
        self.out = ConvModule(
            block_channels[2] * 2,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _make_layer(self,
                    in_channels,
                    out_channels,
                    blocks,
                    stride=1,
                    expand_ratio=6):
        layers = [
            InvertedResidual(
                in_channels,
                out_channels,
                stride,
                expand_ratio,
                norm_cfg=self.norm_cfg)
        ]
        for i in range(1, blocks):
            layers.append(
                InvertedResidual(
                    out_channels,
                    out_channels,
                    1,
                    expand_ratio,
                    norm_cfg=self.norm_cfg))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = torch.cat([x, *self.ppm(x)], dim=1)
        x = self.out(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module.

    Args:
        higher_in_channels (int): Number of input channels of the
            higher-resolution branch.
        lower_in_channels (int): Number of input channels of the
            lower-resolution branch.
        out_channels (int): Number of output channels.
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False
    """

    def __init__(self,
                 higher_in_channels,
                 lower_in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super(FeatureFusionModule, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.dwconv = ConvModule(
            lower_in_channels,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.conv_lower_res = ConvModule(
            out_channels,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.conv_higher_res = ConvModule(
            higher_in_channels,
            out_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = resize(
            lower_res_feature,
            size=higher_res_feature.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


@BACKBONES.register_module()
class FastSCNN(nn.Module):
    """Fast-SCNN Backbone.

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        downsample_dw_channels (tuple[int]): Number of output channels after
            the first conv layer & the second conv layer in
            Learning-To-Downsample (LTD) module.
            Default: (32, 48).
        global_in_channels (int): Number of input channels of
            Global Feature Extractor(GFE).
            Equal to number of output channels of LTD.
            Default: 64.
        global_block_channels (tuple[int]): Tuple of integers that describe
            the output channels for each of the MobileNet-v2 bottleneck
            residual blocks in GFE.
            Default: (64, 96, 128).
        global_block_strides (tuple[int]): Tuple of integers
            that describe the strides (downsampling factors) for each of the
            MobileNet-v2 bottleneck residual blocks in GFE.
            Default: (2, 2, 1).
        global_out_channels (int): Number of output channels of GFE.
            Default: 128.
        higher_in_channels (int): Number of input channels of the higher
            resolution branch in FFM.
            Equal to global_in_channels.
            Default: 64.
        lower_in_channels (int): Number of input channels of  the lower
            resolution branch in FFM.
            Equal to global_out_channels.
            Default: 128.
        fusion_out_channels (int): Number of output channels of FFM.
            Default: 128.
        out_indices (tuple): Tuple of indices of list
            [higher_res_features, lower_res_features, fusion_output].
            Often set to (0,1,2) to enable aux. heads.
            Default: (0, 1, 2).
        conv_cfg (dict | None): Config of conv layers. Default: None
        norm_cfg (dict | None): Config of norm layers. Default:
            dict(type='BN')
        act_cfg (dict): Config of activation layers. Default:
            dict(type='ReLU')
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False
    """

    def __init__(self,
                 in_channels=3,
                 downsample_dw_channels=(32, 48),
                 global_in_channels=64,
                 global_block_channels=(64, 96, 128),
                 global_block_strides=(2, 2, 1),
                 global_out_channels=128,
                 higher_in_channels=64,
                 lower_in_channels=128,
                 fusion_out_channels=128,
                 out_indices=(0, 1, 2),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):

        super(FastSCNN, self).__init__()
        if global_in_channels != higher_in_channels:
            raise AssertionError('Global Input Channels must be the same \
                                 with Higher Input Channels!')
        elif global_out_channels != lower_in_channels:
            raise AssertionError('Global Output Channels must be the same \
                                with Lower Input Channels!')

        self.in_channels = in_channels
        self.downsample_dw_channels1 = downsample_dw_channels[0]
        self.downsample_dw_channels2 = downsample_dw_channels[1]
        self.global_in_channels = global_in_channels
        self.global_block_channels = global_block_channels
        self.global_block_strides = global_block_strides
        self.global_out_channels = global_out_channels
        self.higher_in_channels = higher_in_channels
        self.lower_in_channels = lower_in_channels
        self.fusion_out_channels = fusion_out_channels
        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.learning_to_downsample = LearningToDownsample(
            in_channels,
            downsample_dw_channels,
            global_in_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_feature_extractor = GlobalFeatureExtractor(
            global_in_channels,
            global_block_channels,
            global_out_channels,
            strides=self.global_block_strides,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.feature_fusion = FeatureFusionModule(
            higher_in_channels,
            lower_in_channels,
            fusion_out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        fusion_output = self.feature_fusion(higher_res_features,
                                            lower_res_features)

        outs = [higher_res_features, lower_res_features, fusion_output]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)
