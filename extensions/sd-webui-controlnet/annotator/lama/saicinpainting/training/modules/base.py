import abc
from typing import Tuple, List

import torch
import torch.nn as nn

from annotator.lama.saicinpainting.training.modules.depthwise_sep_conv import DepthWiseSeperableConv
from annotator.lama.saicinpainting.training.modules.multidilated_conv import MultidilatedConv


class BaseDiscriminator(nn.Module):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Predict scores and get intermediate activations. Useful for feature matching loss
        :return tuple (scores, list of intermediate activations)
        """
        raise NotImplemented()


def get_conv_block_ctor(kind='default'):
    if not isinstance(kind, str):
        return kind
    if kind == 'default':
        return nn.Conv2d
    if kind == 'depthwise':
        return DepthWiseSeperableConv   
    if kind == 'multidilated':
        return MultidilatedConv
    raise ValueError(f'Unknown convolutional block kind {kind}')


def get_norm_layer(kind='bn'):
    if not isinstance(kind, str):
        return kind
    if kind == 'bn':
        return nn.BatchNorm2d
    if kind == 'in':
        return nn.InstanceNorm2d
    raise ValueError(f'Unknown norm block kind {kind}')


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class SimpleMultiStepGenerator(nn.Module):
    def __init__(self, steps: List[nn.Module]):
        super().__init__()
        self.steps = nn.ModuleList(steps)

    def forward(self, x):
        cur_in = x
        outs = []
        for step in self.steps:
            cur_out = step(cur_in)
            outs.append(cur_out)
            cur_in = torch.cat((cur_in, cur_out), dim=1)
        return torch.cat(outs[::-1], dim=1)

def deconv_factory(kind, ngf, mult, norm_layer, activation, max_features):
    if kind == 'convtranspose':
        return [nn.ConvTranspose2d(min(max_features, ngf * mult), 
                    min(max_features, int(ngf * mult / 2)), 
                    kernel_size=3, stride=2, padding=1, output_padding=1),
                    norm_layer(min(max_features, int(ngf * mult / 2))), activation]
    elif kind == 'bilinear':
        return [nn.Upsample(scale_factor=2, mode='bilinear'),
                DepthWiseSeperableConv(min(max_features, ngf * mult), 
                    min(max_features, int(ngf * mult / 2)), 
                    kernel_size=3, stride=1, padding=1), 
                norm_layer(min(max_features, int(ngf * mult / 2))), activation]
    else:
        raise Exception(f"Invalid deconv kind: {kind}")