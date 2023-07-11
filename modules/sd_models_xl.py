from __future__ import annotations

import torch

import sgm.models.diffusion
import sgm.modules.diffusionmodules.denoiser_scaling
import sgm.modules.diffusionmodules.discretizer
from modules import devices


def get_learned_conditioning(self: sgm.models.diffusion.DiffusionEngine, batch: list[str]):
    for embedder in self.conditioner.embedders:
        embedder.ucg_rate = 0.0

    c = self.conditioner({'txt': batch})

    return c


def apply_model(self: sgm.models.diffusion.DiffusionEngine, x, t, cond):
    return self.model(x, t, cond)


def extend_sdxl(model):
    dtype = next(model.model.diffusion_model.parameters()).dtype
    model.model.diffusion_model.dtype = dtype
    model.model.conditioning_key = 'crossattn'

    model.cond_stage_model = [x for x in model.conditioner.embedders if type(x).__name__ == 'FrozenOpenCLIPEmbedder'][0]
    model.cond_stage_key = model.cond_stage_model.input_key

    model.parameterization = "v" if isinstance(model.denoiser.scaling, sgm.modules.diffusionmodules.denoiser_scaling.VScaling) else "eps"

    discretization = sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization()
    model.alphas_cumprod = torch.asarray(discretization.alphas_cumprod, device=devices.device, dtype=dtype)


sgm.models.diffusion.DiffusionEngine.get_learned_conditioning = get_learned_conditioning
sgm.models.diffusion.DiffusionEngine.apply_model = apply_model

