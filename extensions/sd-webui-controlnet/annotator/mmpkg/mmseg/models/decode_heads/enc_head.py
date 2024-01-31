import torch
import torch.nn as nn
import torch.nn.functional as F
from annotator.mmpkg.mmcv.cnn import ConvModule, build_norm_layer

from annotator.mmpkg.mmseg.ops import Encoding, resize
from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead


class EncModule(nn.Module):
    """Encoding Module used in EncNet.

    Args:
        in_channels (int): Input channels.
        num_codes (int): Number of code words.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, in_channels, num_codes, conv_cfg, norm_cfg, act_cfg):
        super(EncModule, self).__init__()
        self.encoding_project = ConvModule(
            in_channels,
            in_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # TODO: resolve this hack
        # change to 1d
        if norm_cfg is not None:
            encoding_norm_cfg = norm_cfg.copy()
            if encoding_norm_cfg['type'] in ['BN', 'IN']:
                encoding_norm_cfg['type'] += '1d'
            else:
                encoding_norm_cfg['type'] = encoding_norm_cfg['type'].replace(
                    '2d', '1d')
        else:
            # fallback to BN1d
            encoding_norm_cfg = dict(type='BN1d')
        self.encoding = nn.Sequential(
            Encoding(channels=in_channels, num_codes=num_codes),
            build_norm_layer(encoding_norm_cfg, num_codes)[1],
            nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        """Forward function."""
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)
        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        y = gamma.view(batch_size, channels, 1, 1)
        output = F.relu_(x + x * y)
        return encoding_feat, output


@HEADS.register_module()
class EncHead(BaseDecodeHead):
    """Context Encoding for Semantic Segmentation.

    This head is the implementation of `EncNet
    <https://arxiv.org/abs/1803.08904>`_.

    Args:
        num_codes (int): Number of code words. Default: 32.
        use_se_loss (bool): Whether use Semantic Encoding Loss (SE-loss) to
            regularize the training. Default: True.
        add_lateral (bool): Whether use lateral connection to fuse features.
            Default: False.
        loss_se_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss', use_sigmoid=True).
    """

    def __init__(self,
                 num_codes=32,
                 use_se_loss=True,
                 add_lateral=False,
                 loss_se_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.2),
                 **kwargs):
        super(EncHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.use_se_loss = use_se_loss
        self.add_lateral = add_lateral
        self.num_codes = num_codes
        self.bottleneck = ConvModule(
            self.in_channels[-1],
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if add_lateral:
            self.lateral_convs = nn.ModuleList()
            for in_channels in self.in_channels[:-1]:  # skip the last one
                self.lateral_convs.append(
                    ConvModule(
                        in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.fusion = ConvModule(
                len(self.in_channels) * self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        self.enc_module = EncModule(
            self.channels,
            num_codes=num_codes,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if self.use_se_loss:
            self.loss_se_decode = build_loss(loss_se_decode)
            self.se_layer = nn.Linear(self.channels, self.num_classes)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        feat = self.bottleneck(inputs[-1])
        if self.add_lateral:
            laterals = [
                resize(
                    lateral_conv(inputs[i]),
                    size=feat.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]
            feat = self.fusion(torch.cat([feat, *laterals], 1))
        encode_feat, output = self.enc_module(feat)
        output = self.cls_seg(output)
        if self.use_se_loss:
            se_output = self.se_layer(encode_feat)
            return output, se_output
        else:
            return output

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, ignore se_loss."""
        if self.use_se_loss:
            return self.forward(inputs)[0]
        else:
            return self.forward(inputs)

    @staticmethod
    def _convert_to_onehot_labels(seg_label, num_classes):
        """Convert segmentation label to onehot.

        Args:
            seg_label (Tensor): Segmentation label of shape (N, H, W).
            num_classes (int): Number of classes.

        Returns:
            Tensor: Onehot labels of shape (N, num_classes).
        """

        batch_size = seg_label.size(0)
        onehot_labels = seg_label.new_zeros((batch_size, num_classes))
        for i in range(batch_size):
            hist = seg_label[i].float().histc(
                bins=num_classes, min=0, max=num_classes - 1)
            onehot_labels[i] = hist > 0
        return onehot_labels

    def losses(self, seg_logit, seg_label):
        """Compute segmentation and semantic encoding loss."""
        seg_logit, se_seg_logit = seg_logit
        loss = dict()
        loss.update(super(EncHead, self).losses(seg_logit, seg_label))
        se_loss = self.loss_se_decode(
            se_seg_logit,
            self._convert_to_onehot_labels(seg_label, self.num_classes))
        loss['loss_se'] = se_loss
        return loss
