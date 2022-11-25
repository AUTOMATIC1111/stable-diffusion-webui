import torch

from einops import repeat
from omegaconf import ListConfig

import ldm.models.diffusion.ddpm

from ldm.models.diffusion.ddpm import LatentDiffusion


# =================================================================================================
# Monkey patch LatentInpaintDiffusion to load the checkpoint with a proper config.
# Adapted from:
# https://github.com/runwayml/stable-diffusion/blob/main/ldm/models/diffusion/ddpm.py
# =================================================================================================

@torch.no_grad()
def get_unconditional_conditioning(self, batch_size, null_label=None):
    if null_label is not None:
        xc = null_label
        if isinstance(xc, ListConfig):
            xc = list(xc)
        if isinstance(xc, dict) or isinstance(xc, list):
            c = self.get_learned_conditioning(xc)
        else:
            if hasattr(xc, "to"):
                xc = xc.to(self.device)
            c = self.get_learned_conditioning(xc)
    else:
        # todo: get null label from cond_stage_model
        raise NotImplementedError()
    c = repeat(c, "1 ... -> b ...", b=batch_size).to(self.device)
    return c


class LatentInpaintDiffusion(LatentDiffusion):
    def __init__(
        self,
        concat_keys=("mask", "masked_image"),
        masked_image_key="masked_image",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys


def should_hijack_inpainting(checkpoint_info):
    return "inpainting" in str(checkpoint_info.filename).lower() and not "inpainting" in checkpoint_info.config.lower()


def do_inpainting_hijack():
    ldm.models.diffusion.ddpm.get_unconditional_conditioning = get_unconditional_conditioning
    ldm.models.diffusion.ddpm.LatentInpaintDiffusion = LatentInpaintDiffusion
