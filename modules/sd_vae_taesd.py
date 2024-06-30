"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""
import os
import torch
import torch.nn as nn

from modules import devices, paths_internal, shared

sd_vae_taesd_models = {}


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    @staticmethod
    def forward(x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def decoder(latent_channels=4):
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


def encoder(latent_channels=4):
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, latent_channels),
    )


class TAESDDecoder(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, decoder_path="taesd_decoder.pth", latent_channels=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()

        if latent_channels is None:
            latent_channels = 16 if "taesd3" in str(decoder_path) else 4

        self.decoder = decoder(latent_channels)
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location='cpu' if devices.device.type != 'cuda' else None))


class TAESDEncoder(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", latent_channels=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()

        if latent_channels is None:
            latent_channels = 16 if "taesd3" in str(encoder_path) else 4

        self.encoder = encoder(latent_channels)
        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location='cpu' if devices.device.type != 'cuda' else None))


def download_model(model_path, model_url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        print(f'Downloading TAESD model to: {model_path}')
        torch.hub.download_url_to_file(model_url, model_path)


def decoder_model():
    if shared.sd_model.is_sd3:
        model_name = "taesd3_decoder.pth"
    elif shared.sd_model.is_sdxl:
        model_name = "taesdxl_decoder.pth"
    else:
        model_name = "taesd_decoder.pth"

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, 'https://github.com/madebyollin/taesd/raw/main/' + model_name)

        if os.path.exists(model_path):
            loaded_model = TAESDDecoder(model_path)
            loaded_model.eval()
            loaded_model.to(devices.device, devices.dtype)
            sd_vae_taesd_models[model_name] = loaded_model
        else:
            raise FileNotFoundError('TAESD model not found')

    return loaded_model.decoder


def encoder_model():
    if shared.sd_model.is_sd3:
        model_name = "taesd3_encoder.pth"
    elif shared.sd_model.is_sdxl:
        model_name = "taesdxl_encoder.pth"
    else:
        model_name = "taesd_encoder.pth"

    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, 'https://github.com/madebyollin/taesd/raw/main/' + model_name)

        if os.path.exists(model_path):
            loaded_model = TAESDEncoder(model_path)
            loaded_model.eval()
            loaded_model.to(devices.device, devices.dtype)
            sd_vae_taesd_models[model_name] = loaded_model
        else:
            raise FileNotFoundError('TAESD model not found')

    return loaded_model.encoder
