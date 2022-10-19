import torch
import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import ListConfig

from types import MethodType

import ldm.models.diffusion.ddpm
import ldm.models.diffusion.ddim

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler, noise_like

# =================================================================================================
# Monkey patch DDIMSampler methods from RunwayML repo directly.
# Adapted from:
# https://github.com/runwayml/stable-diffusion/blob/main/ldm/models/diffusion/ddim.py
# =================================================================================================
@torch.no_grad()
def sample(
    self,
    S,
    batch_size,
    shape,
    conditioning=None,
    callback=None,
    normals_sequence=None,
    img_callback=None,
    quantize_x0=False,
    eta=0.,
    mask=None,
    x0=None,
    temperature=1.,
    noise_dropout=0.,
    score_corrector=None,
    corrector_kwargs=None,
    verbose=True,
    x_T=None,
    log_every_t=100,
    unconditional_guidance_scale=1.,
    unconditional_conditioning=None,
    # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
    **kwargs
    ):
    if conditioning is not None:
        if isinstance(conditioning, dict):
            ctmp = conditioning[list(conditioning.keys())[0]]
            while isinstance(ctmp, list):
                ctmp = elf.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

        if self.image_mask is not None:
            conditioning_mask = np.array(self.image_mask.convert("L"))
            conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
            conditioning_mask = torch.from_numpy(conditioning_mask[None, None])

            # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
            conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = torch.ones(1, 1, *image.shape[-2:])

        # Create another latent image, this time with a masked version of the original input.
        conditioning_mask = conditioning_mask.to(image.device)
        conditioning_image = image * (1.0 - conditioning_mask)
        conditioning_image = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(conditioning_image))

        # Create the concatenated conditioning tensor to be fed to `c_concat`
        conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=self.init_latent.shape[-2:])
        conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
        self.image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
        self.image_conditioning = self.image_conditioning.to(shared.device).type(self.sd_model.dtype)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        x = create_random_tensors([opctmp[0]
            cbs = ctmp.shape[0]
            if cbs != batch_size:
                print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
        else:
            if conditioning.shape[0] != batch_size:
                print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

    self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
    # sampling
    C, H, W = shape
    size = (batch_size, C, H, W)
    print(f'Data shape for DDIM sampling is {size}, eta {eta}')

    samples, intermediates = self.ddim_sampling(conditioning, size,
                                                callback=callback,
                                                img_callback=img_callback,
                                                quantize_denoised=quantize_x0,
                                                mask=mask, x0=x0,
                                                ddim_use_original_steps=False,
                                                noise_dropout=noise_dropout,
                                                temperature=temperature,
                                                score_corrector=score_corrector,
                                                corrector_kwargs=corrector_kwargs,
                                                x_T=x_T,
                                                log_every_t=log_every_t,
                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                unconditional_conditioning=unconditional_conditioning,
                                                )
    return samples, intermediates


@torch.no_grad()
def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None):
    b, *_, device = *x.shape, x.device

    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        e_t = self.model.apply_model(x, t, c)
    else:
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        if isinstance(c, dict):
            assert isinstance(unconditional_conditioning, dict)
            c_in = dict()
            for k in c:
                if isinstance(c[k], list):
                    c_in[k] = [
                        torch.cat([unconditional_conditioning[k][i], c[k][i]])
                        for i in range(len(c[k]))
                    ]
                else:
                    c_in[k] = torch.cat([unconditional_conditioning[k], c[k]])
        else:
            c_in = torch.cat([unconditional_conditioning, c])
        e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

    if score_corrector is not None:
        assert self.model.parameterization == "eps"
        e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

    alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    # current prediction for x_0
    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    if quantize_denoised:
        pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    if noise_dropout > 0.:
        noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    return x_prev, pred_x0


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
    return str(checkpoint_info.filename).endswith("inpainting.ckpt") and not checkpoint_info.config.endswith("inpainting.yaml")

def do_inpainting_hijack():
    ldm.models.diffusion.ddpm.get_unconditional_conditioning = get_unconditional_conditioning
    ldm.models.diffusion.ddpm.LatentInpaintDiffusion = LatentInpaintDiffusion
    ldm.models.diffusion.ddim.DDIMSampler.p_sample_ddim = p_sample_ddim
    ldm.models.diffusion.ddim.DDIMSampler.sample = sample