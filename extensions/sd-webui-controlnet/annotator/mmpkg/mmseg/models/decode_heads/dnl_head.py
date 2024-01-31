import torch
from annotator.mmpkg.mmcv.cnn import NonLocal2d
from torch import nn

from ..builder import HEADS
from .fcn_head import FCNHead


class DisentangledNonLocal2d(NonLocal2d):
    """Disentangled Non-Local Blocks.

    Args:
        temperature (float): Temperature to adjust attention. Default: 0.05
    """

    def __init__(self, *arg, temperature, **kwargs):
        super().__init__(*arg, **kwargs)
        self.temperature = temperature
        self.conv_mask = nn.Conv2d(self.in_channels, 1, kernel_size=1)

    def embedded_gaussian(self, theta_x, phi_x):
        """Embedded gaussian with temperature."""

        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight /= self.temperature
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def forward(self, x):
        # x: [N, C, H, W]
        n = x.size(0)

        # g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        # subtract mean
        theta_x -= theta_x.mean(dim=-2, keepdim=True)
        phi_x -= phi_x.mean(dim=-1, keepdim=True)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])

        # unary_mask: [N, 1, HxW]
        unary_mask = self.conv_mask(x)
        unary_mask = unary_mask.view(n, 1, -1)
        unary_mask = unary_mask.softmax(dim=-1)
        # unary_x: [N, 1, C]
        unary_x = torch.matmul(unary_mask, g_x)
        # unary_x: [N, C, 1, 1]
        unary_x = unary_x.permute(0, 2, 1).contiguous().reshape(
            n, self.inter_channels, 1, 1)

        output = x + self.conv_out(y + unary_x)

        return output


@HEADS.register_module()
class DNLHead(FCNHead):
    """Disentangled Non-Local Neural Networks.

    This head is the implementation of `DNLNet
    <https://arxiv.org/abs/2006.06668>`_.

    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: False.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian.'.
        temperature (float): Temperature to adjust attention. Default: 0.05
    """

    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 temperature=0.05,
                 **kwargs):
        super(DNLHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.temperature = temperature
        self.dnl_block = DisentangledNonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            temperature=self.temperature)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.dnl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
