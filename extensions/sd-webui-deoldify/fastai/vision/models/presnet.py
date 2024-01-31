from pdb import set_trace
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['PResNet', 'presnet18', 'presnet34', 'presnet50', 'presnet101', 'presnet152']

act_fn = nn.ReLU

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)
    for l in m.children(): init_cnn(l)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def conv_layer(conv_1st, ni, nf, ks=3, stride=1, zero_bn=False, bias=False):
    bn = nn.BatchNorm2d(nf if conv_1st else ni)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    res = [act_fn(), bn]
    cn = conv(ni, nf, ks, stride=stride, bias=bias)
    res.insert(0 if conv_1st else 2, cn)
    return nn.Sequential(*res)

def conv_act(*args, **kwargs): return conv_layer(True , *args, **kwargs)
def act_conv(*args, **kwargs): return conv_layer(False, *args, **kwargs)

class BasicBlock(Module):
    expansion = 1

    def __init__(self, ni, nf, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = act_conv(ni, nf, stride=stride)
        self.conv2 = act_conv(nf, nf, zero_bn=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        return x

class Bottleneck(Module):
    expansion = 4

    def __init__(self, ni, nf, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = act_conv(ni, nf, 1)
        self.conv2 = act_conv(nf, nf, stride=stride)
        self.conv3 = act_conv(nf, nf*self.expansion, 1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += identity
        return x

class PResNet(Module):

    def __init__(self, block, layers, num_classes=1000):
        self.ni = 64
        super().__init__()
        self.conv1 = conv_act(3, 16, stride=2)
        self.conv2 = conv_act(16, 32)
        self.conv3 = conv_act(32, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        ni = 512*block.expansion
        self.avgpool = nn.Sequential(
            act_fn(), nn.BatchNorm2d(ni), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(ni, num_classes)

        init_cnn(self)

    def _make_layer(self, block, nf, blocks, stride=1):
        downsample = None
        if stride != 1 or self.ni != nf*block.expansion:
            layers = [act_fn(), nn.BatchNorm2d(self.ni),
                      nn.AvgPool2d(kernel_size=2)] if stride==2 else []
            layers.append(conv(self.ni, nf*block.expansion))
            downsample = nn.Sequential(*layers)

        layers = [block(self.ni, nf, stride, downsample)]
        self.ni = nf*block.expansion
        for i in range(1, blocks): layers.append(block(self.ni, nf))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

model_urls = dict(presnet34='presnet34', presnet50='presnet50')

def presnet(block, n_layers, name, pre=False, **kwargs):
    model = PResNet(block, n_layers, **kwargs)
    #if pre: model.load_state_dict(model_zoo.load_url(model_urls[name]))
    if pre: model.load_state_dict(torch.load(model_urls[name]))
    return model

def presnet18(pretrained=False, **kwargs):
    return presnet(BasicBlock, [2, 2, 2, 2], 'presnet18', pre=pretrained, **kwargs)

def presnet34(pretrained=False, **kwargs):
    return presnet(BasicBlock, [3, 4, 6, 3], 'presnet34', pre=pretrained, **kwargs)

def presnet50(pretrained=False, **kwargs):
    return presnet(Bottleneck, [3, 4, 6, 3], 'presnet50', pre=pretrained, **kwargs)

def presnet101(pretrained=False, **kwargs):
    return presnet(Bottleneck, [3, 4, 23, 3], 'presnet101', pre=pretrained, **kwargs)

def presnet152(pretrained=False, **kwargs):
    return presnet(Bottleneck, [3, 8, 36, 3], 'presnet152', pre=pretrained, **kwargs)

