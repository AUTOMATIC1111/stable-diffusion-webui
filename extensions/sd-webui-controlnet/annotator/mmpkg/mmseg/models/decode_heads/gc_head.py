import torch
from annotator.mmpkg.mmcv.cnn import ContextBlock

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class GCHead(FCNHead):
    """GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond.

    This head is the implementation of `GCNet
    <https://arxiv.org/abs/1904.11492>`_.

    Args:
        ratio (float): Multiplier of channels ratio. Default: 1/4.
        pooling_type (str): The pooling type of context aggregation.
            Options are 'att', 'avg'. Default: 'avg'.
        fusion_types (tuple[str]): The fusion type for feature fusion.
            Options are 'channel_add', 'channel_mul'. Default: ('channel_add',)
    """

    def __init__(self,
                 ratio=1 / 4.,
                 pooling_type='att',
                 fusion_types=('channel_add', ),
                 **kwargs):
        super(GCHead, self).__init__(num_convs=2, **kwargs)
        self.ratio = ratio
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.gc_block = ContextBlock(
            in_channels=self.channels,
            ratio=self.ratio,
            pooling_type=self.pooling_type,
            fusion_types=self.fusion_types)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.gc_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
