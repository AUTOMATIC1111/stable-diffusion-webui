import os

import torch
from torch import nn
from modules import devices, paths

sd_vae_approx_model = None


class VAEApprox(nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, (7, 7))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 32, (3, 3))
        self.conv6 = nn.Conv2d(32, 16, (3, 3))
        self.conv7 = nn.Conv2d(16, 8, (3, 3))
        self.conv8 = nn.Conv2d(8, 3, (3, 3))

    def forward(self, x):
        extra = 11
        x = nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = nn.functional.pad(x, (extra, extra, extra, extra))

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, ]:
            x = layer(x)
            x = nn.functional.leaky_relu(x, 0.1)

        return x


def model():
    global sd_vae_approx_model

    if sd_vae_approx_model is None:
        sd_vae_approx_model = VAEApprox()
        sd_vae_approx_model.load_state_dict(torch.load(os.path.join(paths.models_path, "VAE-approx", "model.pt"), map_location='cpu' if devices.device.type != 'cuda' else None))
        sd_vae_approx_model.eval()
        sd_vae_approx_model.to(devices.device, devices.dtype)

    return sd_vae_approx_model


def cheap_approximation(sample):
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2

    coefs = torch.tensor([
        [0.298, 0.207, 0.208],
        [0.187, 0.286, 0.173],
        [-0.158, 0.189, 0.264],
        [-0.184, -0.271, -0.473],
    ]).to(sample.device)

    x_sample = torch.einsum("lxy,lr -> rxy", sample, coefs)

    return x_sample
