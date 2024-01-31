import timm
import torch
import torch.nn as nn
import numpy as np

from .utils import activations, get_activation, Transpose


def forward_levit(pretrained, x):
    pretrained.model.forward_features(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]

    layer_1 = pretrained.act_postprocess1(layer_1)
    layer_2 = pretrained.act_postprocess2(layer_2)
    layer_3 = pretrained.act_postprocess3(layer_3)

    return layer_1, layer_2, layer_3


def _make_levit_backbone(
        model,
        hooks=[3, 11, 21],
        patch_grid=[14, 14]
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))

    pretrained.activations = activations

    patch_grid_size = np.array(patch_grid, dtype=int)

    pretrained.act_postprocess1 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size(patch_grid_size.tolist()))
    )
    pretrained.act_postprocess2 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((np.ceil(patch_grid_size / 2).astype(int)).tolist()))
    )
    pretrained.act_postprocess3 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((np.ceil(patch_grid_size / 4).astype(int)).tolist()))
    )

    return pretrained


class ConvTransposeNorm(nn.Sequential):
    """
    Modification of
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/levit.py: ConvNorm
    such that ConvTranspose2d is used instead of Conv2d.
    """

    def __init__(
            self, in_chs, out_chs, kernel_size=1, stride=1, pad=0, dilation=1,
            groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c',
                        nn.ConvTranspose2d(in_chs, out_chs, kernel_size, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_chs))

        nn.init.constant_(self.bn.weight, bn_weight_init)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.ConvTranspose2d(
            w.size(1), w.size(0), w.shape[2:], stride=self.c.stride,
            padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def stem_b4_transpose(in_chs, out_chs, activation):
    """
    Modification of
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/levit.py: stem_b16
    such that ConvTranspose2d is used instead of Conv2d and stem is also reduced to the half.
    """
    return nn.Sequential(
        ConvTransposeNorm(in_chs, out_chs, 3, 2, 1),
        activation(),
        ConvTransposeNorm(out_chs, out_chs // 2, 3, 2, 1),
        activation())


def _make_pretrained_levit_384(pretrained, hooks=None):
    model = timm.create_model("levit_384", pretrained=pretrained)

    hooks = [3, 11, 21] if hooks == None else hooks
    return _make_levit_backbone(
        model,
        hooks=hooks
    )
