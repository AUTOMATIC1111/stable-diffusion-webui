import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from ...torch_core import Module


__all__ = ['XResNet', 'xresnet18', 'xresnet34_2', 'xresnet50_2', 'xresnet101', 'xresnet152']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv2d(ni, nf, stride):
    return nn.Sequential(nn.Conv2d(ni, nf, kernel_size=3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(nf), nn.ReLU(inplace=True))

class XResNet(Module):

    def __init__(self, block, layers, c_out=1000):
        self.inplanes = 64
        super(XResNet, self).__init__()
        self.conv1 = conv2d(3, 32, 2)
        self.conv2 = conv2d(32, 32, 1)
        self.conv3 = conv2d(32, 64, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, c_out)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock): m.bn2.weight = nn.Parameter(torch.zeros_like(m.bn2.weight))
            if isinstance(m, Bottleneck): m.bn3.weight = nn.Parameter(torch.zeros_like(m.bn3.weight))
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride==2: layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            layers += [
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion) ]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(block(self.inplanes, planes))
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


def xresnet18(pretrained=False, **kwargs):
    """Constructs a XResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = XResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['xresnet18']))
    return model


def xresnet34_2(pretrained=False, **kwargs):
    """Constructs a XResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = XResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['xresnet34']))
    return model


def xresnet50_2(pretrained=False, **kwargs):
    """Constructs a XResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = XResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['xresnet50']))
    return model


def xresnet101(pretrained=False, **kwargs):
    """Constructs a XResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = XResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['xresnet101']))
    return model


def xresnet152(pretrained=False, **kwargs):
    """Constructs a XResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = XResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained: model.load_state_dict(model_zoo.load_url(model_urls['xresnet152']))
    return model

