from collections import namedtuple
import numpy as np
import torch
import tqdm
from PIL import Image

import k_diffusion.sampling
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms

from modules.shared import opts, cmd_opts, state
import modules.shared as shared


SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases'])

samplers_k_diffusion = [
    ('Euler a', 'sample_euler_ancestral', ['k_euler_a']),
    ('Euler', 'sample_euler', ['k_euler']),
    ('LMS', 'sample_lms', ['k_lms']),
    ('Heun', 'sample_heun', ['k_heun']),
    ('DPM2', 'sample_dpm_2', ['k_dpm_2']),
    ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a']),
]

samplers_data_k_diffusion = [
    SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases)
    for label, funcname, aliases in samplers_k_diffusion
    if hasattr(k_diffusion.sampling, funcname)
]

samplers = [
    *samplers_data_k_diffusion,
    SamplerData('DDIM', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.ddim.DDIMSampler, model), []),
    SamplerData('PLMS', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.plms.PLMSSampler, model), []),
]
samplers_for_img2img = [x for x in samplers if x.name != 'PLMS']


def sample_to_image(samples):
    x_sample = shared.sd_model.decode_first_stage(samples[0:1].type(shared.sd_model.dtype))[0]
    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)
    return Image.fromarray(x_sample)


def store_latent(decoded):
    state.current_latent = decoded

    if opts.show_progress_every_n_steps > 0 and shared.state.sampling_step % opts.show_progress_every_n_steps == 0:
        if not shared.parallel_processing_allowed:
            shared.state.current_image = sample_to_image(decoded)


def p_sample_ddim_hook(sampler_wrapper, x_dec, cond, ts, *args, **kwargs):
    if sampler_wrapper.mask is not None:
        img_orig = sampler_wrapper.sampler.model.q_sample(sampler_wrapper.init_latent, ts)
        x_dec = img_orig * sampler_wrapper.mask + sampler_wrapper.nmask * x_dec

    res = sampler_wrapper.orig_p_sample_ddim(x_dec, cond, ts, *args, **kwargs)

    if sampler_wrapper.mask is not None:
        store_latent(sampler_wrapper.init_latent * sampler_wrapper.mask + sampler_wrapper.nmask * res[1])
    else:
        store_latent(res[1])

    return res


def extended_tdqm(sequence, *args, desc=None, **kwargs):
    state.sampling_steps = len(sequence)
    state.sampling_step = 0

    for x in tqdm.tqdm(sequence, *args, desc=state.job, file=shared.progress_print_out, **kwargs):
        if state.interrupted:
            break

        yield x

        state.sampling_step += 1
        shared.total_tqdm.update()


ldm.models.diffusion.ddim.tqdm = lambda *args, desc=None, **kwargs: extended_tdqm(*args, desc=desc, **kwargs)
ldm.models.diffusion.plms.tqdm = lambda *args, desc=None, **kwargs: extended_tdqm(*args, desc=desc, **kwargs)


class VanillaStableDiffusionSampler:
    def __init__(self, constructor, sd_model):
        self.sampler = constructor(sd_model)
        self.orig_p_sample_ddim = self.sampler.p_sample_ddim if hasattr(self.sampler, 'p_sample_ddim') else self.sampler.p_sample_plms
        self.mask = None
        self.nmask = None
        self.init_latent = None

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning):
        t_enc = int(min(p.denoising_strength, 0.999) * p.steps)

        # existing code fails with cetain step counts, like 9
        try:
            self.sampler.make_schedule(ddim_num_steps=p.steps, verbose=False)
        except Exception:
            self.sampler.make_schedule(ddim_num_steps=p.steps+1, verbose=False)

        x1 = self.sampler.stochastic_encode(x, torch.tensor([t_enc] * int(x.shape[0])).to(shared.device), noise=noise)

        self.sampler.p_sample_ddim = lambda x_dec, cond, ts, *args, **kwargs: p_sample_ddim_hook(self, x_dec, cond, ts, *args, **kwargs)
        self.mask = p.mask
        self.nmask = p.nmask
        self.init_latent = p.init_latent

        samples = self.sampler.decode(x1, conditioning, t_enc, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning)

        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning):
        for fieldname in ['p_sample_ddim', 'p_sample_plms']:
            if hasattr(self.sampler, fieldname):
                setattr(self.sampler, fieldname, lambda x_dec, cond, ts, *args, **kwargs: p_sample_ddim_hook(self, x_dec, cond, ts, *args, **kwargs))
        self.mask = None
        self.nmask = None
        self.init_latent = None

        # existing code fails with cetin step counts, like 9
        try:
            samples_ddim, _ = self.sampler.sample(S=p.steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x)
        except Exception:
            samples_ddim, _ = self.sampler.sample(S=p.steps+1, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x)

        return samples_ddim


class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None

    def forward(self, x, sigma, uncond, cond, cond_scale):
        if shared.batch_cond_uncond:
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            denoised = uncond + (cond - uncond) * cond_scale
        else:
            uncond = self.inner_model(x, sigma, cond=uncond)
            cond = self.inner_model(x, sigma, cond=cond)
            denoised = uncond + (cond - uncond) * cond_scale

        if self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        return denoised


def extended_trange(count, *args, **kwargs):
    state.sampling_steps = count
    state.sampling_step = 0

    for x in tqdm.trange(count, *args, desc=state.job, file=shared.progress_print_out, **kwargs):
        if state.interrupted:
            break

        yield x

        state.sampling_step += 1
        shared.total_tqdm.update()


class KDiffusionSampler:
    def __init__(self, funcname, sd_model):
        self.model_wrap = k_diffusion.external.CompVisDenoiser(sd_model)
        self.funcname = funcname
        self.func = getattr(k_diffusion.sampling, self.funcname)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)

    def callback_state(self, d):
        store_latent(d["denoised"])

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning):
        t_enc = int(min(p.denoising_strength, 0.999) * p.steps)
        sigmas = self.model_wrap.get_sigmas(p.steps)

        noise = noise * sigmas[p.steps - t_enc - 1]

        xi = x + noise

        sigma_sched = sigmas[p.steps - t_enc - 1:]

        self.model_wrap_cfg.mask = p.mask
        self.model_wrap_cfg.nmask = p.nmask
        self.model_wrap_cfg.init_latent = p.init_latent

        if hasattr(k_diffusion.sampling, 'trange'):
            k_diffusion.sampling.trange = lambda *args, **kwargs: extended_trange(*args, **kwargs)

        return self.func(self.model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False, callback=self.callback_state)

    def sample(self, p, x, conditioning, unconditional_conditioning):
        sigmas = self.model_wrap.get_sigmas(p.steps)
        x = x * sigmas[0]

        if hasattr(k_diffusion.sampling, 'trange'):
            k_diffusion.sampling.trange = lambda *args, **kwargs: extended_trange(*args, **kwargs)

        samples_ddim = self.func(self.model_wrap_cfg, x, sigmas, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False, callback=self.callback_state)
        return samples_ddim

