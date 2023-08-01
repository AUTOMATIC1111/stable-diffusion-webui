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


def decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, decoder_path="taesd_decoder.pth"):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.decoder = decoder()
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location='cpu' if devices.device.type != 'cuda' else None))

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)


def download_model(model_path, model_url):
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        print(f'Downloading TAESD decoder to: {model_path}')
        torch.hub.download_url_to_file(model_url, model_path)


def model():
    model_name = "taesdxl_decoder.pth" if getattr(shared.sd_model, 'is_sdxl', False) else "taesd_decoder.pth"
    loaded_model = sd_vae_taesd_models.get(model_name)

    if loaded_model is None:
        model_path = os.path.join(paths_internal.models_path, "VAE-taesd", model_name)
        download_model(model_path, 'https://github.com/madebyollin/taesd/raw/main/' + model_name)

        if os.path.exists(model_path):
            loaded_model = TAESD(model_path)
            loaded_model.eval()
            loaded_model.to(devices.device, devices.dtype)
            sd_vae_taesd_models[model_name] = loaded_model
        else:
            raise FileNotFoundError('TAESD model not found')

    return loaded_model.decoder
