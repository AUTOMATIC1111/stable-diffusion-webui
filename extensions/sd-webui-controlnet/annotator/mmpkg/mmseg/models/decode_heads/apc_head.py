import torch
import torch.nn as nn
import torch.nn.functional as F
from annotator.mmpkg.mmcv.cnn import ConvModule

from annotator.mmpkg.mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class ACM(nn.Module):
    """Adaptive Context Module used in APCNet.

    Args:
        pool_scale (int): Pooling scale used in Adaptive Context
            Module to extract region features.
        fusion (bool): Add one conv to fuse residual feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, pool_scale, fusion, in_channels, channels, conv_cfg,
                 norm_cfg, act_cfg):
        super(ACM, self).__init__()
        self.pool_scale = pool_scale
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pooled_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.input_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.global_info = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.gla = nn.Conv2d(self.channels, self.pool_scale**2, 1, 1, 0)

        self.residual_conv = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.fusion:
            self.fusion_conv = ConvModule(
                self.channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        pooled_x = F.adaptive_avg_pool2d(x, self.pool_scale)
        # [batch_size, channels, h, w]
        x = self.input_redu_conv(x)
        # [batch_size, channels, pool_scale, pool_scale]
        pooled_x = self.pooled_redu_conv(pooled_x)
        batch_size = x.size(0)
        # [batch_size, pool_scale * pool_scale, channels]
        pooled_x = pooled_x.view(batch_size, self.channels,
                                 -1).permute(0, 2, 1).contiguous()
        # [batch_size, h * w, pool_scale * pool_scale]
        affinity_matrix = self.gla(x + resize(
            self.global_info(F.adaptive_avg_pool2d(x, 1)), size=x.shape[2:])
                                   ).permute(0, 2, 3, 1).reshape(
                                       batch_size, -1, self.pool_scale**2)
        affinity_matrix = F.sigmoid(affinity_matrix)
        # [batch_size, h * w, channels]
        z_out = torch.matmul(affinity_matrix, pooled_x)
        # [batch_size, channels, h * w]
        z_out = z_out.permute(0, 2, 1).contiguous()
        # [batch_size, channels, h, w]
        z_out = z_out.view(batch_size, self.channels, x.size(2), x.size(3))
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        if self.fusion:
            z_out = self.fusion_conv(z_out)

        return z_out


@HEADS.register_module()
class APCHead(BaseDecodeHead):
    """Adaptive Pyramid Context Network for Semantic Segmentation.

    This head is the implementation of
    `APCNet <https://openaccess.thecvf.com/content_CVPR_2019/papers/\
    He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_\
    CVPR_2019_paper.pdf>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Adaptive Context
            Module. Default: (1, 2, 3, 6).
        fusion (bool): Add one conv to fuse residual feature.
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), fusion=True, **kwargs):
        super(APCHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.fusion = fusion
        acm_modules = []
        for pool_scale in self.pool_scales:
            acm_modules.append(
                ACM(pool_scale,
                    self.fusion,
                    self.in_channels,
                    self.channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.acm_modules = nn.ModuleList(acm_modules)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        acm_outs = [x]
        for acm_module in self.acm_modules:
            acm_outs.append(acm_module(x))
        acm_outs = torch.cat(acm_outs, dim=1)
        output = self.bottleneck(acm_outs)
        output = self.cls_seg(output)
        return output
