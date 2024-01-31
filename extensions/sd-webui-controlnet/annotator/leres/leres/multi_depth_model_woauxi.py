from . import network_auxi as network
from .net_tools import get_func
import torch
import torch.nn as nn
from modules import devices

class RelDepthModel(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(RelDepthModel, self).__init__()
        if backbone == 'resnet50':
            encoder = 'resnet50_stride32'
        elif backbone == 'resnext101':
            encoder = 'resnext101_stride32x8d'
        self.depth_model = DepthModel(encoder)

    def inference(self, rgb):
        with torch.no_grad():
            input = rgb.to(self.depth_model.device)
            depth = self.depth_model(input)
            #pred_depth_out = depth - depth.min() + 0.01
            return depth #pred_depth_out


class DepthModel(nn.Module):
    def __init__(self, encoder):
        super(DepthModel, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + encoder
        self.encoder_modules = get_func(backbone)()
        self.decoder_modules = network.Decoder()

    def forward(self, x):
        lateral_out = self.encoder_modules(x)
        out_logit = self.decoder_modules(lateral_out)
        return out_logit