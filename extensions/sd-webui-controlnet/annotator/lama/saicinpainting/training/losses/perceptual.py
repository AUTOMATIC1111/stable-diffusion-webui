import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# from models.ade20k import ModelBuilder
from annotator.lama.saicinpainting.utils import check_and_warn_input_range


IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]


class PerceptualLoss(nn.Module):
    def __init__(self, normalize_inputs=True):
        super(PerceptualLoss, self).__init__()

        self.normalize_inputs = normalize_inputs
        self.mean_ = IMAGENET_MEAN
        self.std_ = IMAGENET_STD

        vgg = torchvision.models.vgg19(pretrained=True).features
        vgg_avg_pooling = []

        for weights in vgg.parameters():
            weights.requires_grad = False

        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)

        self.vgg = nn.Sequential(*vgg_avg_pooling)

    def do_normalize_inputs(self, x):
        return (x - self.mean_.to(x.device)) / self.std_.to(x.device)

    def partial_losses(self, input, target, mask=None):
        check_and_warn_input_range(target, 0, 1, 'PerceptualLoss target in partial_losses')

        # we expect input and target to be in [0, 1] range
        losses = []

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
            features_target = self.do_normalize_inputs(target)
        else:
            features_input = input
            features_target = target

        for layer in self.vgg[:30]:

            features_input = layer(features_input)
            features_target = layer(features_target)

            if layer.__class__.__name__ == 'ReLU':
                loss = F.mse_loss(features_input, features_target, reduction='none')

                if mask is not None:
                    cur_mask = F.interpolate(mask, size=features_input.shape[-2:],
                                             mode='bilinear', align_corners=False)
                    loss = loss * (1 - cur_mask)

                loss = loss.mean(dim=tuple(range(1, len(loss.shape))))
                losses.append(loss)

        return losses

    def forward(self, input, target, mask=None):
        losses = self.partial_losses(input, target, mask=mask)
        return torch.stack(losses).sum(dim=0)

    def get_global_features(self, input):
        check_and_warn_input_range(input, 0, 1, 'PerceptualLoss input in get_global_features')

        if self.normalize_inputs:
            features_input = self.do_normalize_inputs(input)
        else:
            features_input = input

        features_input = self.vgg(features_input)
        return features_input


class ResNetPL(nn.Module):
    def __init__(self, weight=1,
                 weights_path=None, arch_encoder='resnet50dilated', segmentation=True):
        super().__init__()
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
                                             arch_encoder=arch_encoder,
                                             arch_decoder='ppm_deepsup',
                                             fc_dim=2048,
                                             segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)

        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)

        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)

        result = torch.stack([F.mse_loss(cur_pred, cur_target)
                              for cur_pred, cur_target
                              in zip(pred_feats, target_feats)]).sum() * self.weight
        return result
