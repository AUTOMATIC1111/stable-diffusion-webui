import os
import torch

from einops import repeat
from omegaconf import ListConfig

import ldm.models.diffusion.ddpm
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler, noise_like

# =================================================================================================
# Monkey patch DDIMSampler methods from RunwayML repo directly.
# Adapted from:
# https://github.com/runwayml/stable-diffusion/blob/main/ldm/models/diffusion/ddim.py
# =================================================================================================
@torch.no_grad()
def sample_ddim(self,
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
                ctmp = ctmp[0]
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
# Monkey patch PLMSSampler methods.
# This one was not actually patched correctly in the RunwayML repo, but we can replicate the changes.
# Adapted from:
# https://github.com/CompVis/stable-diffusion/blob/main/ldm/models/diffusion/plms.py
# =================================================================================================
@torch.no_grad()
def sample_plms(self,
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
                ctmp = ctmp[0]
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
    print(f'Data shape for PLMS sampling is {size}')

    samples, intermediates = self.plms_sampling(conditioning, size,
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
def p_sample_plms(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                  unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None, dynamic_threshold=None):
    b, *_, device = *x.shape, x.device

    def get_model_output(x, t):
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

        return e_t

    alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

    def get_x_prev_and_pred_x0(e_t, index):
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        if dynamic_threshold is not None:
            pred_x0 = norm_thresholding(pred_x0, dynamic_threshold)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    e_t = get_model_output(x, t)
    if len(old_eps) == 0:
        # Pseudo Improved Euler (2nd order)
        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
        e_t_next = get_model_output(x_prev, t_next)
        e_t_prime = (e_t + e_t_next) / 2
    elif len(old_eps) == 1:
        # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
        e_t_prime = (3 * e_t - old_eps[-1]) / 2
    elif len(old_eps) == 2:
        # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
        e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
    elif len(old_eps) >= 3:
        # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
        e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

    x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

    return x_prev, pred_x0, e_t

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
    ckpt_basename = os.path.basename(checkpoint_info.filename).lower()
    cfg_basename = os.path.basename(checkpoint_info.config).lower()
    return "inpainting" in ckpt_basename and not "inpainting" in cfg_basename


def do_inpainting_hijack():
    # most of this stuff seems to no longer be needed because it is already included into SD2.0
    # p_sample_plms is needed because PLMS can't work with dicts as conditionings
    # this file should be cleaned up later if everything turns out to work fine

    # ldm.models.diffusion.ddpm.get_unconditional_conditioning = get_unconditional_conditioning
    # ldm.models.diffusion.ddpm.LatentInpaintDiffusion = LatentInpaintDiffusion

    # ldm.models.diffusion.ddim.DDIMSampler.p_sample_ddim = p_sample_ddim
    # ldm.models.diffusion.ddim.DDIMSampler.sample = sample_ddim

    ldm.models.diffusion.plms.PLMSSampler.p_sample_plms = p_sample_plms
    # ldm.models.diffusion.plms.PLMSSampler.sample = sample_plms
