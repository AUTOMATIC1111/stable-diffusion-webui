import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from annotator.mmpkg.mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from annotator.mmpkg.mmcv.runner import load_checkpoint
from annotator.mmpkg.mmcv.utils.parrots_wrapper import _BatchNorm

from annotator.mmpkg.mmseg.utils import get_root_logger
from ..builder import BACKBONES


class GlobalContextExtractor(nn.Module):
    """Global Context Extractor for CGNet.

    This class is employed to refine the joint feature of both local feature
    and surrounding context.

    Args:
        channel (int): Number of input feature channels.
        reduction (int): Reductions for global context extractor. Default: 16.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self, channel, reduction=16, with_cp=False):
        super(GlobalContextExtractor, self).__init__()
        self.channel = channel
        self.reduction = reduction
        assert reduction >= 1 and channel >= reduction
        self.with_cp = with_cp
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):

        def _inner_forward(x):
            num_batch, num_channel = x.size()[:2]
            y = self.avg_pool(x).view(num_batch, num_channel)
            y = self.fc(y).view(num_batch, num_channel, 1, 1)
            return x * y

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class ContextGuidedBlock(nn.Module):
    """Context Guided Block for CGNet.

    This class consists of four components: local feature extractor,
    surrounding feature extractor, joint feature extractor and global
    context extractor.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        dilation (int): Dilation rate for surrounding context extractor.
            Default: 2.
        reduction (int): Reduction for global context extractor. Default: 16.
        skip_connect (bool): Add input to output or not. Default: True.
        downsample (bool): Downsample the input to 1/2 or not. Default: False.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation=2,
                 reduction=16,
                 skip_connect=True,
                 downsample=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU'),
                 with_cp=False):
        super(ContextGuidedBlock, self).__init__()
        self.with_cp = with_cp
        self.downsample = downsample

        channels = out_channels if downsample else out_channels // 2
        if 'type' in act_cfg and act_cfg['type'] == 'PReLU':
            act_cfg['num_parameters'] = channels
        kernel_size = 3 if downsample else 1
        stride = 2 if downsample else 1
        padding = (kernel_size - 1) // 2

        self.conv1x1 = ConvModule(
            in_channels,
            channels,
            kernel_size,
            stride,
            padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.f_loc = build_conv_layer(
            conv_cfg,
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False)
        self.f_sur = build_conv_layer(
            conv_cfg,
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            groups=channels,
            dilation=dilation,
            bias=False)

        self.bn = build_norm_layer(norm_cfg, 2 * channels)[1]
        self.activate = nn.PReLU(2 * channels)

        if downsample:
            self.bottleneck = build_conv_layer(
                conv_cfg,
                2 * channels,
                out_channels,
                kernel_size=1,
                bias=False)

        self.skip_connect = skip_connect and not downsample
        self.f_glo = GlobalContextExtractor(out_channels, reduction, with_cp)

    def forward(self, x):

        def _inner_forward(x):
            out = self.conv1x1(x)
            loc = self.f_loc(out)
            sur = self.f_sur(out)

            joi_feat = torch.cat([loc, sur], 1)  # the joint feature
            joi_feat = self.bn(joi_feat)
            joi_feat = self.activate(joi_feat)
            if self.downsample:
                joi_feat = self.bottleneck(joi_feat)  # channel = out_channels
            # f_glo is employed to refine the joint feature
            out = self.f_glo(joi_feat)

            if self.skip_connect:
                return x + out
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class InputInjection(nn.Module):
    """Downsampling module for CGNet."""

    def __init__(self, num_downsampling):
        super(InputInjection, self).__init__()
        self.pool = nn.ModuleList()
        for i in range(num_downsampling):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)
        return x


@BACKBONES.register_module()
class CGNet(nn.Module):
    """CGNet backbone.

    A Light-weight Context Guided Network for Semantic Segmentation
    arXiv: https://arxiv.org/abs/1811.08201

    Args:
        in_channels (int): Number of input image channels. Normally 3.
        num_channels (tuple[int]): Numbers of feature channels at each stages.
            Default: (32, 64, 128).
        num_blocks (tuple[int]): Numbers of CG blocks at stage 1 and stage 2.
            Default: (3, 21).
        dilations (tuple[int]): Dilation rate for surrounding context
            extractors at stage 1 and stage 2. Default: (2, 4).
        reductions (tuple[int]): Reductions for global context extractors at
            stage 1 and stage 2. Default: (8, 16).
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='PReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 num_channels=(32, 64, 128),
                 num_blocks=(3, 21),
                 dilations=(2, 4),
                 reductions=(8, 16),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='PReLU'),
                 norm_eval=False,
                 with_cp=False):

        super(CGNet, self).__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        assert isinstance(self.num_channels, tuple) and len(
            self.num_channels) == 3
        self.num_blocks = num_blocks
        assert isinstance(self.num_blocks, tuple) and len(self.num_blocks) == 2
        self.dilations = dilations
        assert isinstance(self.dilations, tuple) and len(self.dilations) == 2
        self.reductions = reductions
        assert isinstance(self.reductions, tuple) and len(self.reductions) == 2
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if 'type' in self.act_cfg and self.act_cfg['type'] == 'PReLU':
            self.act_cfg['num_parameters'] = num_channels[0]
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        cur_channels = in_channels
        self.stem = nn.ModuleList()
        for i in range(3):
            self.stem.append(
                ConvModule(
                    cur_channels,
                    num_channels[0],
                    3,
                    2 if i == 0 else 1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            cur_channels = num_channels[0]

        self.inject_2x = InputInjection(1)  # down-sample for Input, factor=2
        self.inject_4x = InputInjection(2)  # down-sample for Input, factor=4

        cur_channels += in_channels
        self.norm_prelu_0 = nn.Sequential(
            build_norm_layer(norm_cfg, cur_channels)[1],
            nn.PReLU(cur_channels))

        # stage 1
        self.level1 = nn.ModuleList()
        for i in range(num_blocks[0]):
            self.level1.append(
                ContextGuidedBlock(
                    cur_channels if i == 0 else num_channels[1],
                    num_channels[1],
                    dilations[0],
                    reductions[0],
                    downsample=(i == 0),
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))  # CG block

        cur_channels = 2 * num_channels[1] + in_channels
        self.norm_prelu_1 = nn.Sequential(
            build_norm_layer(norm_cfg, cur_channels)[1],
            nn.PReLU(cur_channels))

        # stage 2
        self.level2 = nn.ModuleList()
        for i in range(num_blocks[1]):
            self.level2.append(
                ContextGuidedBlock(
                    cur_channels if i == 0 else num_channels[2],
                    num_channels[2],
                    dilations[1],
                    reductions[1],
                    downsample=(i == 0),
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp))  # CG block

        cur_channels = 2 * num_channels[2]
        self.norm_prelu_2 = nn.Sequential(
            build_norm_layer(norm_cfg, cur_channels)[1],
            nn.PReLU(cur_channels))

    def forward(self, x):
        output = []

        # stage 0
        inp_2x = self.inject_2x(x)
        inp_4x = self.inject_4x(x)
        for layer in self.stem:
            x = layer(x)
        x = self.norm_prelu_0(torch.cat([x, inp_2x], 1))
        output.append(x)

        # stage 1
        for i, layer in enumerate(self.level1):
            x = layer(x)
            if i == 0:
                down1 = x
        x = self.norm_prelu_1(torch.cat([x, down1, inp_4x], 1))
        output.append(x)

        # stage 2
        for i, layer in enumerate(self.level2):
            x = layer(x)
            if i == 0:
                down2 = x
        x = self.norm_prelu_2(torch.cat([down2, x], 1))
        output.append(x)

        return output

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
                elif isinstance(m, nn.PReLU):
                    constant_init(m, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        """Convert the model into training mode will keeping the normalization
        layer freezed."""
        super(CGNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
