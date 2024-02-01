"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""
import os
from PIL import Image
import torch
import torch.nn as nn
from modules import devices, paths


taesd_models = { 'sd-decoder': None, 'sd-encoder': None, 'sdxl-decoder': None, 'sdxl-encoder': None }


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder():
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 4),
    )

def Decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

class TAESD(nn.Module): # pylint: disable=abstract-method
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", decoder_path="taesd_decoder.pth"):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)


def download_model(model_path):
    model_name = os.path.basename(model_path)
    model_url = f'https://github.com/madebyollin/taesd/raw/main/{model_name}'
    if not os.path.exists(model_path):
        import torch
        from modules.shared import log
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        log.info(f'Downloading TAESD decoder: {model_path}')
        torch.hub.download_url_to_file(model_url, model_path)


def model(model_class = 'sd', model_type = 'decoder'):
    vae = taesd_models[f'{model_class}-{model_type}']
    if vae is None:
        model_path = os.path.join(paths.models_path, "TAESD", f"tae{model_class}_{model_type}.pth")
        download_model(model_path)
        if os.path.exists(model_path):
            from modules.shared import log
            taesd_models[f'{model_class}-{model_type}'] = TAESD(decoder_path=model_path, encoder_path=None) if model_type == 'decoder' else TAESD(encoder_path=model_path, decoder_path=None)
            vae = taesd_models[f'{model_class}-{model_type}']
            vae.eval()
            vae.to(devices.device, devices.dtype_vae)
            log.info(f"Load VAE-TAESD: model={model_path}")
        else:
            raise FileNotFoundError(f'TAESD model not found: {model_path}')
    if vae is None:
        return None
    else:
        return vae.decoder if model_type == 'decoder' else vae.encoder


def decode(latents):
    from modules import shared
    model_class = shared.sd_model_type
    if model_class == 'ldm':
        model_class = 'sd'
    if 'sd' not in model_class:
        shared.log.warning(f'TAESD unsupported model type: {model_class}')
        return Image.new('RGB', (8, 8), color = (0, 0, 0))
    vae = taesd_models[f'{model_class}-decoder']
    if vae is None:
        model_path = os.path.join(paths.models_path, "TAESD", f"tae{model_class}_decoder.pth")
        download_model(model_path)
        if os.path.exists(model_path):
            taesd_models[f'{model_class}-decoder'] = TAESD(decoder_path=model_path, encoder_path=None)
            shared.log.debug(f'VAE load: type=taesd model={model_path}')
            vae = taesd_models[f'{model_class}-decoder']
            vae.decoder.to(devices.device, devices.dtype_vae)
    try:
        with devices.inference_context():
            latents = latents.detach().clone().to(devices.device, devices.dtype_vae)
            if len(latents.shape) == 3:
                latents = latents.unsqueeze(0)
                image = vae.decoder(latents).clamp(0, 1).detach()
                image = 2.0 * image - 1.0 # typical normalized range except for preview which runs denormalization
                return image[0]
            elif len(latents.shape) == 4:
                image = vae.decoder(latents).clamp(0, 1).detach()
                image = 2.0 * image - 1.0 # typical normalized range except for preview which runs denormalization
                return image
            else:
                shared.log.error(f'TAESD decode unsupported latent type: {latents.shape}')
                return latents
    except Exception as e:
        shared.log.error(f'VAE decode taesd: {e}')
        return latents


def encode(image):
    from modules import shared
    model_class = shared.sd_model_type
    if model_class == 'ldm':
        model_class = 'sd'
    if 'sd' not in model_class:
        shared.log.warning(f'TAESD unsupported model type: {model_class}')
        return Image.new('RGB', (8, 8), color = (0, 0, 0))
    vae = taesd_models[f'{model_class}-encoder']
    if vae is None:
        model_path = os.path.join(paths.models_path, "TAESD", f"tae{model_class}_encoder.pth")
        download_model(model_path)
        if os.path.exists(model_path):
            shared.log.debug(f'VAE load: type=taesd model={model_path}')
            taesd_models[f'{model_class}-encoder'] = TAESD(encoder_path=model_path, decoder_path=None)
            vae = taesd_models[f'{model_class}-encoder']
            vae.encoder.to(devices.device, devices.dtype_vae)
    # image = vae.scale_latents(image)
    latents = vae.encoder(image)
    return latents.detach()
