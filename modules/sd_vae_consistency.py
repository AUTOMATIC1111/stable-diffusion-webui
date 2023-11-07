"""
Consistency Decoder
Improved decoding for stable diffusion vaes.

https://github.com/openai/consistencydecoder
"""
import os

from modules import devices, paths_internal, shared
from consistencydecoder import ConsistencyDecoder


sd_vae_consistency_models = None
model_path = os.path.join(paths_internal.models_path, 'consistencydecoder')


def decoder_model():
    global sd_vae_consistency_models
    if getattr(shared.sd_model, 'is_sdxl', False):
        raise NotImplementedError("SDXL is not supported for consistency decoder")
    if sd_vae_consistency_models is not None:
        sd_vae_consistency_models.ckpt.to(devices.device)
        return sd_vae_consistency_models

    loaded_model = ConsistencyDecoder(devices.device, model_path)
    sd_vae_consistency_models = loaded_model
    return loaded_model


def unload():
    global sd_vae_consistency_models
    if sd_vae_consistency_models is not None:
        devices.torch_gc()
        sd_vae_consistency_models.ckpt.to('cpu')
