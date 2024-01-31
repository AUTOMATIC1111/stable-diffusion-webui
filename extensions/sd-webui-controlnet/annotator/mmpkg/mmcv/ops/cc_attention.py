# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from annotator.mmpkg.mmcv.cnn import PLUGIN_LAYERS, Scale


def NEG_INF_DIAG(n, device):
    """Returns a diagonal matrix of size [n, n].

    The diagonal are all "-inf". This is for avoiding calculating the
    overlapped element in the Criss-Cross twice.
    """
    return torch.diag(torch.tensor(float('-inf')).to(device).repeat(n), 0)


@PLUGIN_LAYERS.register_module()
class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module.

    .. note::
        Before v1.3.13, we use a CUDA op. Since v1.3.13, we switch
        to a pure PyTorch and equivalent implementation. For more
        details, please refer to https://github.com/open-mmlab/mmcv/pull/1201.

        Speed comparison for one forward pass

        - Input size: [2,512,97,97]
        - Device: 1 NVIDIA GeForce RTX 2080 Ti

        +-----------------------+---------------+------------+---------------+
        |                       |PyTorch version|CUDA version|Relative speed |
        +=======================+===============+============+===============+
        |with torch.no_grad()   |0.00554402 s   |0.0299619 s |5.4x           |
        +-----------------------+---------------+------------+---------------+
        |no with torch.no_grad()|0.00562803 s   |0.0301349 s |5.4x           |
        +-----------------------+---------------+------------+---------------+

    Args:
        in_channels (int): Channels of the input feature map.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = Scale(0.)
        self.in_channels = in_channels

    def forward(self, x):
        """forward function of Criss-Cross Attention.

        Args:
            x (Tensor): Input feature. \
                shape (batch_size, in_channels, height, width)
        Returns:
            Tensor: Output of the layer, with shape of \
            (batch_size, in_channels, height, width)
        """
        B, C, H, W = x.size()
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        energy_H = torch.einsum('bchw,bciw->bwhi', query, key) + NEG_INF_DIAG(
            H, query.device)
        energy_H = energy_H.transpose(1, 2)
        energy_W = torch.einsum('bchw,bchj->bhwj', query, key)
        attn = F.softmax(
            torch.cat([energy_H, energy_W], dim=-1), dim=-1)  # [B,H,W,(H+W)]
        out = torch.einsum('bciw,bhwi->bchw', value, attn[..., :H])
        out += torch.einsum('bchj,bhwj->bchw', value, attn[..., H:])

        out = self.gamma(out) + x
        out = out.contiguous()

        return out

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels})'
        return s
