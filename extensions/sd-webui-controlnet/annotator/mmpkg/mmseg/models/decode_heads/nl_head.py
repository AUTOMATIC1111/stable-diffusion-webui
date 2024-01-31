import torch
from annotator.mmpkg.mmcv.cnn import NonLocal2d

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class NLHead(FCNHead):
    """Non-local Neural Networks.

    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.

    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: True.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian.'.
    """

    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 **kwargs):
        super(NLHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.nl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
