## code from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
import torch.nn as nn
import math
import torch
import numpy as np
import random
from .custom_trans_v1 import *


def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GELU)):
            pass
        else:
            m.initialize()



class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,padding=1,dilation=1,stride=1,groups=1,\
                tbn=True,trelu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,\
            stride=stride,padding=padding,dilation=dilation,groups=groups)

        self.bn = nn.BatchNorm2d(out_ch)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.GELU()
        self.tbn = tbn
        self.trelu = trelu

    def forward(self, x):
        out = self.conv(x)
        if self.tbn:
            out = self.bn(out)
        if self.trelu:
            out = self.relu(out)
        return out

    def initialize(self):
        weight_init(self)

