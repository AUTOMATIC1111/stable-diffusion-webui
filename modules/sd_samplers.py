from collections import namedtuple
import numpy as np
import torch
import tqdm
from PIL import Image
import inspect
import k_diffusion.sampling
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
from modules import prompt_parser

from modules.shared import opts, cmd_opts, state
import modules.shared as shared


SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])

samplers_k_diffusion = [
    ('Euler a', 'sample_euler_ancestral', ['k_euler_a'], {}),
    ('Euler', 'sample_euler', ['k_euler'], {}),
    ('LMS', 'sample_lms', ['k_lms'], {}),
    ('Heun', 'sample_heun', ['k_heun'], {}),
    ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {}),
    ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {}),
    ('DPM fast', 'sample_dpm_fast', ['k_dpm_fast'], {}),
    ('DPM adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {}),
    ('LMS Karras', 'sample_lms', ['k_lms_ka'], {'scheduler': 'karras'}),
    ('DPM2 Karras', 'sample_dpm_2', ['k_dpm_2_ka'], {'scheduler': 'karras'}),
    ('DPM2 a Karras', 'sample_dpm_2_ancestral', ['k_dpm_2_a_ka'], {'scheduler': 'karras'}),
]

samplers_data_k_diffusion = [
    SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_k_diffusion
    if hasattr(k_diffusion.sampling, funcname)
]

all_samplers = [
    *samplers_data_k_diffusion,
    SamplerData('DDIM', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.ddim.DDIMSampler, model), [], {}),
    SamplerData('PLMS', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.plms.PLMSSampler, model), [], {}),
]

samplers = []
samplers_for_img2img = []


def create_sampler_with_index(list_of_configs, index, model):
    config = list_of_configs[index]
    sampler = config.constructor(model)
    sampler.config = config
    
    return sampler


def set_samplers():
    global samplers, samplers_for_img2img

    hidden = set(opts.hide_samplers)
    hidden_img2img = set(opts.hide_samplers + ['PLMS', 'DPM fast', 'DPM adaptive'])

    samplers = [x for x in all_samplers if x.name not in hidden]
    samplers_for_img2img = [x for x in all_samplers if x.name not in hidden_img2img]


set_samplers()

sampler_extra_params = {
    'sample_euler': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_heun': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_dpm_2': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
}

def setup_img2img_steps(p, steps=None):
    if opts.img2img_fix_steps or steps is not None:
        steps = int((steps or p.steps) / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = p.steps - 1
    else:
        steps = p.steps
        t_enc = int(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc


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



def extended_tdqm(sequence, *args, desc=None, **kwargs):
    state.sampling_steps = len(sequence)
    state.sampling_step = 0

    seq = sequence if cmd_opts.disable_console_progressbars else tqdm.tqdm(sequence, *args, desc=state.job, file=shared.progress_print_out, **kwargs)

    for x in seq:
        if state.interrupted or state.skipped:
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
        self.sampler_noises = None
        self.step = 0
        self.eta = None
        self.default_eta = 0.0
        self.config = None

    def number_of_needed_noises(self, p):
        return 0

    def p_sample_ddim_hook(self, x_dec, cond, ts, unconditional_conditioning, *args, **kwargs):
        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        unconditional_conditioning = prompt_parser.reconstruct_cond_batch(unconditional_conditioning, self.step)

        assert all([len(conds) == 1 for conds in conds_list]), 'composition via AND is not supported for DDIM/PLMS samplers'
        cond = tensor

        # for DDIM, shapes must match, we can't just process cond and uncond independently;
        # filling unconditional_conditioning with repeats of the last vector to match length is
        # not 100% correct but should work well enough
        if unconditional_conditioning.shape[1] < cond.shape[1]:
            last_vector = unconditional_conditioning[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - unconditional_conditioning.shape[1], 1])
            unconditional_conditioning = torch.hstack([unconditional_conditioning, last_vector_repeated])
        elif unconditional_conditioning.shape[1] > cond.shape[1]:
            unconditional_conditioning = unconditional_conditioning[:, :cond.shape[1]]

        if self.mask is not None:
            img_orig = self.sampler.model.q_sample(self.init_latent, ts)
            x_dec = img_orig * self.mask + self.nmask * x_dec

        res = self.orig_p_sample_ddim(x_dec, cond, ts, unconditional_conditioning=unconditional_conditioning, *args, **kwargs)

        if self.mask is not None:
            store_latent(self.init_latent * self.mask + self.nmask * res[1])
        else:
            store_latent(res[1])

        self.step += 1
        return res

    def initialize(self, p):
        self.eta = p.eta if p.eta is not None else opts.eta_ddim

        for fieldname in ['p_sample_ddim', 'p_sample_plms']:
            if hasattr(self.sampler, fieldname):
                setattr(self.sampler, fieldname, self.p_sample_ddim_hook)

        self.mask = p.mask if hasattr(p, 'mask') else None
        self.nmask = p.nmask if hasattr(p, 'nmask') else None

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None):
        steps, t_enc = setup_img2img_steps(p, steps)

        self.initialize(p)

        # existing code fails with certain step counts, like 9
        try:
            self.sampler.make_schedule(ddim_num_steps=steps,  ddim_eta=self.eta, ddim_discretize=p.ddim_discretize, verbose=False)
        except Exception:
            self.sampler.make_schedule(ddim_num_steps=steps+1, ddim_eta=self.eta, ddim_discretize=p.ddim_discretize, verbose=False)

        x1 = self.sampler.stochastic_encode(x, torch.tensor([t_enc] * int(x.shape[0])).to(shared.device), noise=noise)

        self.init_latent = x
        self.step = 0

        samples = self.sampler.decode(x1, conditioning, t_enc, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning)

        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None):
        self.initialize(p)

        self.init_latent = None
        self.step = 0

        steps = steps or p.steps

        # existing code fails with certain step counts, like 9
        try:
            samples_ddim, _ = self.sampler.sample(S=steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x, eta=self.eta)
        except Exception:
            samples_ddim, _ = self.sampler.sample(S=steps+1, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x, eta=self.eta)

        return samples_ddim


class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.step = 0

    def forward(self, x, sigma, uncond, cond, cond_scale):
        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]

        x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
        sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])

        if tensor.shape[1] == uncond.shape[1]:
            cond_in = torch.cat([tensor, uncond])

            if shared.batch_cond_uncond:
                x_out = self.inner_model(x_in, sigma_in, cond=cond_in)
            else:
                x_out = torch.zeros_like(x_in)
                for batch_offset in range(0, x_out.shape[0], batch_size):
                    a = batch_offset
                    b = a + batch_size
                    x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=cond_in[a:b])
        else:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size*2 if shared.batch_cond_uncond else batch_size
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])
                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=tensor[a:b])

            x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond=uncond)

        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)

        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)

        if self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        self.step += 1

        return denoised


def extended_trange(sampler, count, *args, **kwargs):
    state.sampling_steps = count
    state.sampling_step = 0

    seq = range(count) if cmd_opts.disable_console_progressbars else tqdm.trange(count, *args, desc=state.job, file=shared.progress_print_out, **kwargs)

    for x in seq:
        if state.interrupted or state.skipped:
            break

        if sampler.stop_at is not None and x > sampler.stop_at:
            break

        yield x

        state.sampling_step += 1
        shared.total_tqdm.update()


class TorchHijack:
    def __init__(self, kdiff_sampler):
        self.kdiff_sampler = kdiff_sampler

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.kdiff_sampler.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))


class KDiffusionSampler:
    def __init__(self, funcname, sd_model):
        self.model_wrap = k_diffusion.external.CompVisDenoiser(sd_model, quantize=shared.opts.enable_quantization)
        self.funcname = funcname
        self.func = getattr(k_diffusion.sampling, self.funcname)
        self.extra_params = sampler_extra_params.get(funcname, [])
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.sampler_noises = None
        self.sampler_noise_index = 0
        self.stop_at = None
        self.eta = None
        self.default_eta = 1.0
        self.config = None

    def callback_state(self, d):
        store_latent(d["denoised"])

    def number_of_needed_noises(self, p):
        return p.steps

    def randn_like(self, x):
        noise = self.sampler_noises[self.sampler_noise_index] if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises) else None

        if noise is not None and x.shape == noise.shape:
            res = noise
        else:
            res = torch.randn_like(x)

        self.sampler_noise_index += 1
        return res

    def initialize(self, p):
        self.model_wrap_cfg.mask = p.mask if hasattr(p, 'mask') else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, 'nmask') else None
        self.model_wrap.step = 0
        self.sampler_noise_index = 0
        self.eta = p.eta or opts.eta_ancestral

        if hasattr(k_diffusion.sampling, 'trange'):
            k_diffusion.sampling.trange = lambda *args, **kwargs: extended_trange(self, *args, **kwargs)

        if self.sampler_noises is not None:
            k_diffusion.sampling.torch = TorchHijack(self)

        extra_params_kwargs = {}
        for param_name in self.extra_params:
            if hasattr(p, param_name) and param_name in inspect.signature(self.func).parameters:
                extra_params_kwargs[param_name] = getattr(p, param_name)

        if 'eta' in inspect.signature(self.func).parameters:
            extra_params_kwargs['eta'] = self.eta

        return extra_params_kwargs

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None):
        steps, t_enc = setup_img2img_steps(p, steps)

        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif self.config is not None and self.config.options.get('scheduler', None) == 'karras':
            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=0.1, sigma_max=10, device=shared.device)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)

        noise = noise * sigmas[steps - t_enc - 1]
        xi = x + noise

        extra_params_kwargs = self.initialize(p)

        sigma_sched = sigmas[steps - t_enc - 1:]

        self.model_wrap_cfg.init_latent = x

        return self.func(self.model_wrap_cfg, xi, sigma_sched, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False, callback=self.callback_state, **extra_params_kwargs)

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None):
        steps = steps or p.steps

        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif self.config is not None and self.config.options.get('scheduler', None) == 'karras':
            sigmas = k_diffusion.sampling.get_sigmas_karras(n=steps, sigma_min=0.1, sigma_max=10, device=shared.device)
        else:
            sigmas = self.model_wrap.get_sigmas(steps)

        x = x * sigmas[0]

        extra_params_kwargs = self.initialize(p)
        if 'sigma_min' in inspect.signature(self.func).parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[-1].item()
            if 'n' in inspect.signature(self.func).parameters:
                extra_params_kwargs['n'] = steps
        else:
            extra_params_kwargs['sigmas'] = sigmas
        samples = self.func(self.model_wrap_cfg, x, extra_args={'cond': conditioning, 'uncond': unconditional_conditioning, 'cond_scale': p.cfg_scale}, disable=False, callback=self.callback_state, **extra_params_kwargs)
        return samples

