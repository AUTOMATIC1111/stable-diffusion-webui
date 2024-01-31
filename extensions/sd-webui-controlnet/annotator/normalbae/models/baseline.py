import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodules.submodules import UpSampleBN, norm_normalize


# This is the baseline encoder-decoder we used in the ablation study
class NNET(nn.Module):
    def __init__(self, args=None):
        super(NNET, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=4)

    def forward(self, x, **kwargs):
        out = self.decoder(self.encoder(x), **kwargs)

        # Bilinearly upsample the output to match the input resolution
        up_out = F.interpolate(out, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=False)
        
        # L2-normalize the first three channels / ensure positive value for concentration parameters (kappa)
        up_out = norm_normalize(up_out) 
        return up_out

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder]
        for m in modules:
            yield from m.parameters()


# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        basemodel_name = 'tf_efficientnet_b5_ap'
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)

        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


# Decoder (no pixel-wise MLP, no uncertainty-guided sampling)
class Decoder(nn.Module):
    def __init__(self, num_classes=4):
        super(Decoder, self).__init__()
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(skip_input=2048 + 176, output_features=1024)
        self.up2 = UpSampleBN(skip_input=1024 + 64, output_features=512)
        self.up3 = UpSampleBN(skip_input=512 + 40, output_features=256)
        self.up4 = UpSampleBN(skip_input=256 + 24, output_features=128)
        self.conv3 = nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return out


if __name__ == '__main__':
    model = Baseline()
    x = torch.rand(2, 3, 480, 640)
    out = model(x)
    print(out.shape)
