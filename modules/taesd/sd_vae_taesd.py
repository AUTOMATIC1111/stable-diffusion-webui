"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""
import os
from PIL import Image
from modules import devices, paths
from modules.taesd.taesd import TAESD

taesd_models = { 'sd-decoder': None, 'sd-encoder': None, 'sdxl-decoder': None, 'sdxl-encoder': None }

def download_model(model_path):
    model_name = os.path.basename(model_path)
    model_url = f'https://github.com/madebyollin/taesd/raw/main/{model_name}'
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        from modules.shared import log
        log.info(f'Downloading TAESD decoder: {model_path}')
        import torch
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
            vae = taesd_models[f'{model_class}-decoder']
            vae.to(devices.device, devices.dtype_vae)
    latents.to(devices.device, devices.dtype_vae)
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
            taesd_models[f'{model_class}-encoder'] = TAESD(encoder_path=model_path, decoder_path=None)
            vae = taesd_models[f'{model_class}-encoder']
            vae.to(devices.device, devices.dtype_vae)
    # image = vae.scale_latents(image)
    latents = vae.encoder(image)
    return latents.detach()
