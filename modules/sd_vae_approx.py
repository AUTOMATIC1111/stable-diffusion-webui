import os

import torch
from torch import nn
from modules import devices, paths, shared

sd_vae_approx_models = {}


class VAEApprox(nn.Module):
    def __init__(self, latent_channels=4):
        super(VAEApprox, self).__init__()
        self.conv1 = nn.Conv2d(latent_channels, 8, (7, 7))
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


def download_model(model_path, model_url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        print(f'Downloading VAEApprox model to: {model_path}')
        torch.hub.download_url_to_file(model_url, model_path)


def model():
    if shared.sd_model.is_sd3:
        model_name = "vaeapprox-sd3.pt"
    elif shared.sd_model.is_sdxl:
        model_name = "vaeapprox-sdxl.pt"
    elif shared.sd_model.is_flux1:
        model_name = "vaeapprox-sd3.pt"
    else:
        model_name = "model.pt"

    loaded_model = sd_vae_approx_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths.models_path, "VAE-approx", model_name)
        if not os.path.exists(model_path):
            model_path = os.path.join(paths.script_path, "models", "VAE-approx", model_name)

        if not os.path.exists(model_path):
            model_path = os.path.join(paths.models_path, "VAE-approx", model_name)
            download_model(model_path, 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/download/v1.0.0-pre/' + model_name)

        loaded_model = VAEApprox(latent_channels=shared.sd_model.latent_channels)
        loaded_model.load_state_dict(torch.load(model_path, map_location='cpu' if devices.device.type != 'cuda' else None))
        loaded_model.eval()
        loaded_model.to(devices.device, devices.dtype)
        sd_vae_approx_models[model_name] = loaded_model

    return loaded_model


def cheap_approximation(sample):
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2

    if shared.sd_model.is_sd3:
        coeffs = [
            [-0.0645,  0.0177,  0.1052], [ 0.0028,  0.0312,  0.0650],
            [ 0.1848,  0.0762,  0.0360], [ 0.0944,  0.0360,  0.0889],
            [ 0.0897,  0.0506, -0.0364], [-0.0020,  0.1203,  0.0284],
            [ 0.0855,  0.0118,  0.0283], [-0.0539,  0.0658,  0.1047],
            [-0.0057,  0.0116,  0.0700], [-0.0412,  0.0281, -0.0039],
            [ 0.1106,  0.1171,  0.1220], [-0.0248,  0.0682, -0.0481],
            [ 0.0815,  0.0846,  0.1207], [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456], [-0.1418, -0.1457, -0.1259],
        ]
    elif shared.sd_model.is_flux1:
        coeffs = [
            # from comfy
            [-0.0404,  0.0159,  0.0609], [ 0.0043,  0.0298,  0.0850],
            [ 0.0328, -0.0749, -0.0503], [-0.0245,  0.0085,  0.0549],
            [ 0.0966,  0.0894,  0.0530], [ 0.0035,  0.0399,  0.0123],
            [ 0.0583,  0.1184,  0.1262], [-0.0191, -0.0206, -0.0306],
            [-0.0324,  0.0055,  0.1001], [ 0.0955,  0.0659, -0.0545],
            [-0.0504,  0.0231, -0.0013], [ 0.0500, -0.0008, -0.0088],
            [ 0.0982,  0.0941,  0.0976], [-0.1233, -0.0280, -0.0897],
            [-0.0005, -0.0530, -0.0020], [-0.1273, -0.0932, -0.0680],
        ]
    elif shared.sd_model.is_sdxl:
        coeffs = [
            [ 0.3448,  0.4168,  0.4395],
            [-0.1953, -0.0290,  0.0250],
            [ 0.1074,  0.0886, -0.0163],
            [-0.3730, -0.2499, -0.2088],
        ]
    else:
        coeffs = [
            [ 0.298,  0.207,  0.208],
            [ 0.187,  0.286,  0.173],
            [-0.158,  0.189,  0.264],
            [-0.184, -0.271, -0.473],
        ]

    coefs = torch.tensor(coeffs).to(sample.device)

    x_sample = torch.einsum("...lxy,lr -> ...rxy", sample, coefs)

    return x_sample
