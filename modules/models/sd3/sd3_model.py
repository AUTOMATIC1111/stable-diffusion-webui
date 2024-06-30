import contextlib

import torch

import k_diffusion
from modules.models.sd3.sd3_impls import BaseModel, SDVAE, SD3LatentFormat
from modules.models.sd3.sd3_cond import SD3Cond

from modules import shared, devices


class SD3Denoiser(k_diffusion.external.DiscreteSchedule):
    def __init__(self, inner_model, sigmas):
        super().__init__(sigmas, quantize=shared.opts.enable_quantization)
        self.inner_model = inner_model

    def forward(self, input, sigma, **kwargs):
        return self.inner_model.apply_model(input, sigma, **kwargs)


class SD3Inferencer(torch.nn.Module):
    def __init__(self, state_dict, shift=3, use_ema=False):
        super().__init__()

        self.shift = shift

        with torch.no_grad():
            self.model = BaseModel(shift=shift, state_dict=state_dict, prefix="model.diffusion_model.", device="cpu", dtype=devices.dtype)
            self.first_stage_model = SDVAE(device="cpu", dtype=devices.dtype_vae)
            self.first_stage_model.dtype = self.model.diffusion_model.dtype

        self.alphas_cumprod = 1 / (self.model.model_sampling.sigmas ** 2 + 1)

        self.text_encoders = SD3Cond()
        self.cond_stage_key = 'txt'

        self.parameterization = "eps"
        self.model.conditioning_key = "crossattn"

        self.latent_format = SD3LatentFormat()
        self.latent_channels = 16

    @property
    def cond_stage_model(self):
        return self.text_encoders

    def before_load_weights(self, state_dict):
        self.cond_stage_model.before_load_weights(state_dict)

    def ema_scope(self):
        return contextlib.nullcontext()

    def get_learned_conditioning(self, batch: list[str]):
        return self.cond_stage_model(batch)

    def apply_model(self, x, t, cond):
        return self.model(x, t, c_crossattn=cond['crossattn'], y=cond['vector'])

    def decode_first_stage(self, latent):
        latent = self.latent_format.process_out(latent)
        return self.first_stage_model.decode(latent)

    def encode_first_stage(self, image):
        latent = self.first_stage_model.encode(image)
        return self.latent_format.process_in(latent)

    def get_first_stage_encoding(self, x):
        return x

    def create_denoiser(self):
        return SD3Denoiser(self, self.model.model_sampling.sigmas)

    def medvram_fields(self):
        return [
            (self, 'first_stage_model'),
            (self, 'text_encoders'),
            (self, 'model'),
        ]

    def add_noise_to_latent(self, x, noise, amount):
        return x * (1 - amount) + noise * amount

    def fix_dimensions(self, width, height):
        return width // 16 * 16, height // 16 * 16
