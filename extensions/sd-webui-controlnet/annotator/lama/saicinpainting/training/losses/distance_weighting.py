import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from annotator.lama.saicinpainting.training.losses.perceptual import IMAGENET_STD, IMAGENET_MEAN


def dummy_distance_weighter(real_img, pred_img, mask):
    return mask


def get_gauss_kernel(kernel_size, width_factor=1):
    coords = torch.stack(torch.meshgrid(torch.arange(kernel_size),
                                        torch.arange(kernel_size)),
                         dim=0).float()
    diff = torch.exp(-((coords - kernel_size // 2) ** 2).sum(0) / kernel_size / width_factor)
    diff /= diff.sum()
    return diff


class BlurMask(nn.Module):
    def __init__(self, kernel_size=5, width_factor=1):
        super().__init__()
        self.filter = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, padding_mode='replicate', bias=False)
        self.filter.weight.data.copy_(get_gauss_kernel(kernel_size, width_factor=width_factor))

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            result = self.filter(mask) * mask
            return result


class EmulatedEDTMask(nn.Module):
    def __init__(self, dilate_kernel_size=5, blur_kernel_size=5, width_factor=1):
        super().__init__()
        self.dilate_filter = nn.Conv2d(1, 1, dilate_kernel_size, padding=dilate_kernel_size// 2, padding_mode='replicate',
                                       bias=False)
        self.dilate_filter.weight.data.copy_(torch.ones(1, 1, dilate_kernel_size, dilate_kernel_size, dtype=torch.float))
        self.blur_filter = nn.Conv2d(1, 1, blur_kernel_size, padding=blur_kernel_size // 2, padding_mode='replicate', bias=False)
        self.blur_filter.weight.data.copy_(get_gauss_kernel(blur_kernel_size, width_factor=width_factor))

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            known_mask = 1 - mask
            dilated_known_mask = (self.dilate_filter(known_mask) > 1).float()
            result = self.blur_filter(1 - dilated_known_mask) * mask
            return result


class PropagatePerceptualSim(nn.Module):
    def __init__(self, level=2, max_iters=10, temperature=500, erode_mask_size=3):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        vgg_avg_pooling = []

        for weights in vgg.parameters():
            weights.requires_grad = False

        cur_level_i = 0
        for module in vgg.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg_avg_pooling.append(module)
                if module.__class__.__name__ == 'ReLU':
                    cur_level_i += 1
                if cur_level_i == level:
                    break

        self.features = nn.Sequential(*vgg_avg_pooling)

        self.max_iters = max_iters
        self.temperature = temperature
        self.do_erode = erode_mask_size > 0
        if self.do_erode:
            self.erode_mask = nn.Conv2d(1, 1, erode_mask_size, padding=erode_mask_size // 2, bias=False)
            self.erode_mask.weight.data.fill_(1)

    def forward(self, real_img, pred_img, mask):
        with torch.no_grad():
            real_img = (real_img - IMAGENET_MEAN.to(real_img)) / IMAGENET_STD.to(real_img)
            real_feats = self.features(real_img)

            vertical_sim = torch.exp(-(real_feats[:, :, 1:] - real_feats[:, :, :-1]).pow(2).sum(1, keepdim=True)
                                     / self.temperature)
            horizontal_sim = torch.exp(-(real_feats[:, :, :, 1:] - real_feats[:, :, :, :-1]).pow(2).sum(1, keepdim=True)
                                       / self.temperature)

            mask_scaled = F.interpolate(mask, size=real_feats.shape[-2:], mode='bilinear', align_corners=False)
            if self.do_erode:
                mask_scaled = (self.erode_mask(mask_scaled) > 1).float()

            cur_knowness = 1 - mask_scaled

            for iter_i in range(self.max_iters):
                new_top_knowness = F.pad(cur_knowness[:, :, :-1] * vertical_sim, (0, 0, 1, 0), mode='replicate')
                new_bottom_knowness = F.pad(cur_knowness[:, :, 1:] * vertical_sim, (0, 0, 0, 1), mode='replicate')

                new_left_knowness = F.pad(cur_knowness[:, :, :, :-1] * horizontal_sim, (1, 0, 0, 0), mode='replicate')
                new_right_knowness = F.pad(cur_knowness[:, :, :, 1:] * horizontal_sim, (0, 1, 0, 0), mode='replicate')

                new_knowness = torch.stack([new_top_knowness, new_bottom_knowness,
                                            new_left_knowness, new_right_knowness],
                                           dim=0).max(0).values

                cur_knowness = torch.max(cur_knowness, new_knowness)

            cur_knowness = F.interpolate(cur_knowness, size=mask.shape[-2:], mode='bilinear')
            result = torch.min(mask, 1 - cur_knowness)

            return result


def make_mask_distance_weighter(kind='none', **kwargs):
    if kind == 'none':
        return dummy_distance_weighter
    if kind == 'blur':
        return BlurMask(**kwargs)
    if kind == 'edt':
        return EmulatedEDTMask(**kwargs)
    if kind == 'pps':
        return PropagatePerceptualSim(**kwargs)
    raise ValueError(f'Unknown mask distance weighter kind {kind}')
