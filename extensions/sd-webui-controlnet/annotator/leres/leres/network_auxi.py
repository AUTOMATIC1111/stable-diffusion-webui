import torch
import torch.nn as nn
import torch.nn.init as init

from . import Resnet, Resnext_torch


def resnet50_stride32():
    return DepthNet(backbone='resnet', depth=50, upfactors=[2, 2, 2, 2])

def resnext101_stride32x8d():
    return DepthNet(backbone='resnext101_32x8d', depth=101, upfactors=[2, 2, 2, 2])


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.inchannels =  [256, 512, 1024, 2048]
        self.midchannels = [256, 256, 256, 512]
        self.upfactors = [2,2,2,2]
        self.outchannels = 1

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)
        
        self.ffm2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], upfactor=self.upfactors[2])
        self.ffm1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], upfactor=self.upfactors[1])
        self.ffm0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])
        
        self.outconv = AO(inchannels=self.midchannels[0], outchannels=self.outchannels, upfactor=2)
        self._init_params()
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def forward(self, features):
        x_32x = self.conv(features[3])  # 1/32
        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)  # 1/16

        x_8 = self.ffm2(features[2], x_16)  # 1/8
        x_4 = self.ffm1(features[1], x_8)  # 1/4
        x_2 = self.ffm0(features[0], x_4)  # 1/2
        #-----------------------------------------
        x = self.outconv(x_2)  # original size
        return x

class DepthNet(nn.Module):
    __factory = {
        18: Resnet.resnet18,
        34: Resnet.resnet34,
        50: Resnet.resnet50,
        101: Resnet.resnet101,
        152: Resnet.resnet152
    }
    def __init__(self,
                 backbone='resnet',
                 depth=50,
                 upfactors=[2, 2, 2, 2]):
        super(DepthNet, self).__init__()
        self.backbone = backbone
        self.depth = depth
        self.pretrained = False
        self.inchannels = [256, 512, 1024, 2048]
        self.midchannels = [256, 256, 256, 512]
        self.upfactors = upfactors
        self.outchannels = 1

        # Build model
        if self.backbone == 'resnet':
            if self.depth not in DepthNet.__factory:
                raise KeyError("Unsupported depth:", self.depth)
            self.encoder = DepthNet.__factory[depth](pretrained=self.pretrained)
        elif self.backbone == 'resnext101_32x8d':
            self.encoder = Resnext_torch.resnext101_32x8d(pretrained=self.pretrained)
        else:
            self.encoder = Resnext_torch.resnext101(pretrained=self.pretrained)

    def forward(self, x):
        x = self.encoder(x)  # 1/32, 1/16, 1/8, 1/4
        return x


class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels
        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1,
                               bias=True)
        # NN.BatchNorm2d
        self.conv_branch = nn.Sequential(nn.ReLU(inplace=True), \
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3,
                                                   padding=1, stride=1, bias=True), \
                                         nn.BatchNorm2d(num_features=self.mid), \
                                         nn.ReLU(inplace=True), \
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3,
                                                   padding=1, stride=1, bias=True))
        self.relu = nn.ReLU(inplace=True)

        self.init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.relu(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ATA(nn.Module):
    def __init__(self, inchannels, reduction=8):
        super(ATA, self).__init__()
        self.inchannels = inchannels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.inchannels * 2, self.inchannels // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.inchannels // reduction, self.inchannels),
                                nn.Sigmoid())
        self.init_params()

    def forward(self, low_x, high_x):
        n, c, _, _ = low_x.size()
        x = torch.cat([low_x, high_x], 1)
        x = self.avg_pool(x)
        x = x.view(n, -1)
        x = self.fc(x).view(n, c, 1, 1)
        x = low_x * x + high_x

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                # init.normal(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                # init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        # self.ata = ATA(inchannels = self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True)

        self.init_params()

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        x = x + high_x
        x = self.ftb2(x)
        x = self.upsample(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, upfactor=2):
        super(AO, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.BatchNorm2d(num_features=self.inchannels // 2), \
            nn.ReLU(inplace=True), \
            nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True))

        self.init_params()

    def forward(self, x):
        x = self.adapt_conv(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)



# ==============================================================================================================


class ResidualConv(nn.Module):
    def __init__(self, inchannels):
        super(ResidualConv, self).__init__()
        # NN.BatchNorm2d
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(num_features=inchannels),
            nn.ReLU(inplace=False),
            # nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1, stride=1, groups=inchannels,bias=True),
            # nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=1, padding=0, stride=1, groups=1,bias=True)
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels / 2, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=inchannels / 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=inchannels / 2, out_channels=inchannels, kernel_size=3, padding=1, stride=1,
                      bias=False)
        )
        self.init_params()

    def forward(self, x):
        x = self.conv(x) + x
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class FeatureFusion(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(FeatureFusion, self).__init__()
        self.conv = ResidualConv(inchannels=inchannels)
        # NN.BatchNorm2d
        self.up = nn.Sequential(ResidualConv(inchannels=inchannels),
                                nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,
                                                   stride=2, padding=1, output_padding=1),
                                nn.BatchNorm2d(num_features=outchannels),
                                nn.ReLU(inplace=True))

    def forward(self, lowfeat, highfeat):
        return self.up(highfeat + self.conv(lowfeat))

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class SenceUnderstand(nn.Module):
    def __init__(self, channels):
        super(SenceUnderstand, self).__init__()
        self.channels = channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Sequential(nn.Linear(512 * 8 * 8, self.channels),
                                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True))
        self.initial_params()

    def forward(self, x):
        n, c, h, w = x.size()
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(n, -1)
        x = self.fc(x)
        x = x.view(n, self.channels, 1, 1)
        x = self.conv2(x)
        x = x.repeat(1, 1, h, w)
        return x

    def initial_params(self, dev=0.01):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, dev)


if __name__ == '__main__':
    net = DepthNet(depth=50, pretrained=True)
    print(net)
    inputs = torch.ones(4,3,128,128)
    out = net(inputs)
    print(out.size())

