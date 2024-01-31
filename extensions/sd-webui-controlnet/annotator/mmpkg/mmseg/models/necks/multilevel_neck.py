import torch.nn as nn
import torch.nn.functional as F
from annotator.mmpkg.mmcv.cnn import ConvModule

from ..builder import NECKS


@NECKS.register_module()
class MultiLevelNeck(nn.Module):
    """MultiLevelNeck.

    A neck structure connect vit backbone and decoder_heads.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[int]): Scale factors for each input feature map.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=[0.5, 1, 2, 4],
                 norm_cfg=None,
                 act_cfg=None):
        super(MultiLevelNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        for _ in range(self.num_outs):
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        print(inputs[0].shape)
        inputs = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # for len(inputs) not equal to self.num_outs
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]
        outs = []
        for i in range(self.num_outs):
            x_resize = F.interpolate(
                inputs[i], scale_factor=self.scales[i], mode='bilinear')
            outs.append(self.convs[i](x_resize))
        return tuple(outs)
