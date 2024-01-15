import sys
import time
import inspect
from collections import deque
import torch
from modules import prompt_parser
from modules import devices
from modules import sd_samplers_common
import modules.shared as shared
from modules.script_callbacks import CFGDenoiserParams, cfg_denoiser_callback
from modules.script_callbacks import CFGDenoisedParams, cfg_denoised_callback
from modules.script_callbacks import AfterCFGCallbackParams, cfg_after_cfg_callback
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback


# deal with k-diffusion imports
k_sampling = None
try:
    import k_diffusion # pylint: disable=wrong-import-order
    k_sampling = k_diffusion.sampling
except ImportError:
    pass
try:
    if k_sampling is None:
        import importlib
        k_diffusion = importlib.import_module('modules.k-diffusion.k_diffusion')
        k_sampling = k_diffusion.sampling
except Exception:
    pass
if k_sampling is None:
    shared.log.info(f'Path search: {sys.path}')
    shared.log.error("Module not found: k-diffusion")
    sys.exit(1)


samplers_k_diffusion = [
    ('Euler', 'sample_euler', ['k_euler'], {"scheduler": "default"}),
    ('Euler a', 'sample_euler_ancestral', ['k_euler_a', 'k_euler_ancestral'], {"scheduler": "default", "brownian_noise": False}),
    ('Heun', 'sample_heun', ['k_heun'], {"scheduler": "default"}),
    ('LMS', 'sample_lms', ['k_lms'], {"scheduler": "default"}),
    ('DPM Adaptive', 'sample_dpm_adaptive', ['k_dpm_ad'], {"scheduler": "default", "brownian_noise": False}),
    ('DPM Fast', 'sample_dpm_fast', ['k_dpm_fast'], {"scheduler": "default", "brownian_noise": False}),
    ('DPM2', 'sample_dpm_2', ['k_dpm_2'], {'discard_next_to_last_sigma': True, "second_order": True, "scheduler": "default", "brownian_noise": False}),
    ('DPM2 a', 'sample_dpm_2_ancestral', ['k_dpm_2_a'], {'discard_next_to_last_sigma': True, "second_order": True, "scheduler": "default", "brownian_noise": False}),
    ('DPM++ 2M', 'sample_dpmpp_2m', ['k_dpmpp_2m'], {"scheduler": "default", "brownian_noise": False}),
    ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde'], {'discard_next_to_last_sigma': True, "scheduler": "default", "brownian_noise": False}),
    ('DPM++ 2M SDE Heun', 'sample_dpmpp_2m_sde', ['k_dpmpp_2m_sde_heun'], {"solver_type": "heun", "scheduler": "default", "brownian_noise": False}),
    ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', ['k_dpmpp_2s_a'], {"second_order": True, "scheduler": "default", "brownian_noise": False}),
    ('DPM++ SDE', 'sample_dpmpp_sde', ['k_dpmpp_sde'], {"second_order": True, "scheduler": "default", "brownian_noise": False}),
    ('DPM++ 3M SDE', 'sample_dpmpp_3m_sde', ['k_dpmpp_3m_sde'], {'discard_next_to_last_sigma': True, "scheduler": "default", "brownian_noise": False}),
]

samplers_data_k_diffusion = [
    sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_k_diffusion
    if hasattr(k_sampling, funcname)
]

sampler_extra_params = {
    'sample_euler': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_heun': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
    'sample_dpm_2': ['s_churn', 's_tmin', 's_tmax', 's_noise'],
}


class CFGDenoiser(torch.nn.Module):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.step = 0
        self.image_cfg_scale = None

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)
        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)
        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)
        return denoised

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        if shared.state.interrupted or shared.state.skipped:
            raise sd_samplers_common.InterruptedException
        if shared.state.paused:
            shared.log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise sd_samplers_common.InterruptedException
                time.sleep(0.1)
        # at self.image_cfg_scale == 1.0 produced results for edit model are the same as with normal sampling,
        # so is_edit_model is set to False to support AND composition.
        is_edit_model = (shared.sd_model is not None) and hasattr(shared.sd_model, 'cond_stage_key') and (shared.sd_model.cond_stage_key == "edit") and (self.image_cfg_scale is not None) and (self.image_cfg_scale != 1.0)
        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)
        assert not is_edit_model or all(len(conds) == 1 for conds in conds_list), "AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)"
        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]
        if shared.sd_model.model.conditioning_key == "crossattn-adm":
            image_uncond = torch.zeros_like(image_cond)
            make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": c_crossattn, "c_adm": c_adm} # pylint: disable=C3001
        else:
            image_uncond = image_cond
            make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": c_crossattn, "c_concat": [c_concat]} # pylint: disable=C3001
        if not is_edit_model:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond])
        else:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond] + [torch.zeros_like(self.init_latent)])
        denoiser_params = CFGDenoiserParams(x_in, image_cond_in, sigma_in, shared.state.sampling_step, shared.state.sampling_steps, tensor, uncond)
        cfg_denoiser_callback(denoiser_params)
        x_in = denoiser_params.x
        image_cond_in = denoiser_params.image_cond
        sigma_in = denoiser_params.sigma
        tensor = denoiser_params.text_cond
        uncond = denoiser_params.text_uncond
        skip_uncond = False
        # alternating uncond allows for higher thresholds without the quality loss normally expected from raising it
        if self.step % 2 and s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model:
            skip_uncond = True
            x_in = x_in[:-batch_size]
            sigma_in = sigma_in[:-batch_size]

        if tensor.shape[1] == uncond.shape[1] or skip_uncond:
            if is_edit_model:
                cond_in = torch.cat([tensor, uncond, uncond])
            elif skip_uncond:
                cond_in = tensor
            else:
                cond_in = torch.cat([tensor, uncond])
            """
            adjusted_cond_scale = cond_scale # Adjusted cond_scale for uncond
            last_uncond_steps = max(0, state.sampling_steps - 2) # Determine the last two steps before uncond stops
            if self.step >= last_uncond_steps: # Check if we're in the last two steps before uncond stops
                adjusted_cond_scale *= 1.5 # Apply uncond with 150% cond_scale
            else:
                if (self.step - last_uncond_steps) % 3 == 0: # Check if it's one of every three steps after uncond stops
                    adjusted_cond_scale *= 1.5 # Apply uncond with 150% cond_scale
            """
            if shared.batch_cond_uncond:
                x_out = self.inner_model(x_in, sigma_in, cond=make_condition_dict([cond_in], image_cond_in))
            else:
                x_out = torch.zeros_like(x_in)
                for batch_offset in range(0, x_out.shape[0], batch_size):
                    a = batch_offset
                    b = a + batch_size
                    x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=make_condition_dict([cond_in[a:b]], image_cond_in[a:b]))
        else:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size*2 if shared.batch_cond_uncond else batch_size
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])
                if not is_edit_model:
                    c_crossattn = [tensor[a:b]]
                else:
                    c_crossattn = torch.cat([tensor[a:b]], uncond)
                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=make_condition_dict(c_crossattn, image_cond_in[a:b]))
            if not skip_uncond:
                x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond=make_condition_dict([uncond], image_cond_in[-uncond.shape[0]:]))
        denoised_image_indexes = [x[0][0] for x in conds_list]
        if skip_uncond:
            fake_uncond = torch.cat([x_out[i:i+1] for i in denoised_image_indexes])
            x_out = torch.cat([x_out, fake_uncond])  # we skipped uncond denoising, so we put cond-denoised image to where the uncond-denoised image should be
        denoised_params = CFGDenoisedParams(x_out, shared.state.sampling_step, shared.state.sampling_steps, self.inner_model)
        cfg_denoised_callback(denoised_params)
        devices.test_for_nans(x_out, "unet")
        if shared.opts.live_preview_content == "Prompt":
            sd_samplers_common.store_latent(torch.cat([x_out[i:i+1] for i in denoised_image_indexes]))
        elif shared.opts.live_preview_content == "Negative prompt":
            sd_samplers_common.store_latent(x_out[-uncond.shape[0]:])
        if is_edit_model:
            denoised = self.combine_denoised_for_edit_model(x_out, cond_scale)
        elif skip_uncond:
            denoised = self.combine_denoised(x_out, conds_list, uncond, 1.0)
        else:
            denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale)
        if self.mask is not None:
            if devices.backend == "directml":
                self.init_latent = self.init_latent.float()
                denoised = self.init_latent * self.mask + self.nmask * denoised
                self.init_latent = self.init_latent.half()
            else:
                denoised = self.init_latent * self.mask + self.nmask * denoised
        after_cfg_callback_params = AfterCFGCallbackParams(denoised, shared.state.sampling_step, shared.state.sampling_steps)
        cfg_after_cfg_callback(after_cfg_callback_params)
        denoised = after_cfg_callback_params.x
        self.step += 1
        return denoised


class TorchHijack:
    def __init__(self, sampler_noises):
        # Using a deque to efficiently receive the sampler_noises in the same order as the previous index-based
        # implementation.
        self.sampler_noises = deque(sampler_noises)

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.randn_like
        if hasattr(torch, item):
            return getattr(torch, item)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        if self.sampler_noises:
            noise = self.sampler_noises.popleft()
            if noise.shape == x.shape:
                return noise
        if x.device.type == 'mps':
            return torch.randn_like(x, device=devices.cpu).to(x.device)
        else:
            return torch.randn_like(x)


class KDiffusionSampler:
    def __init__(self, funcname, sd_model):
        denoiser = k_diffusion.external.CompVisVDenoiser if sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser
        self.model_wrap = denoiser(sd_model, quantize=shared.opts.enable_quantization)
        self.funcname = funcname
        self.func = getattr(k_sampling, self.funcname)
        self.extra_params = sampler_extra_params.get(funcname, [])
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config = None  # set by the function calling the constructor
        self.last_latent = None
        self.s_min_uncond = None
        self.conditioning_key = sd_model.model.conditioning_key

    def callback_state(self, d):
        step = d['i']
        latent = d["denoised"]
        if shared.opts.live_preview_content == "Combined":
            sd_samplers_common.store_latent(latent)
        self.last_latent = latent
        if self.stop_at is not None and step > self.stop_at:
            raise sd_samplers_common.InterruptedException
        shared.state.sampling_step = step

    def launch_sampling(self, steps, func):
        shared.state.sampling_steps = steps
        shared.state.sampling_step = 0
        try:
            return func()
        except sd_samplers_common.InterruptedException:
            return self.last_latent

    def number_of_needed_noises(self, p):
        return p.steps

    def initialize(self, p):
        if self.config.options.get('brownian_noise', None) is not None:
            self.config.options['brownian_noise'] = shared.opts.data.get('schedulers_brownian_noise', False)
        if self.config.options.get('scheduler', None) is not None:
            self.config.options['scheduler'] = shared.opts.data.get('schedulers_sigma', None)
        if p is None:
            return {}
        self.model_wrap_cfg.mask = p.mask if hasattr(p, 'mask') else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, 'nmask') else None
        self.model_wrap_cfg.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.eta = p.eta if p.eta is not None else shared.opts.scheduler_eta
        self.s_min_uncond = getattr(p, 's_min_uncond', 0.0)
        k_sampling.torch = TorchHijack(self.sampler_noises if self.sampler_noises is not None else [])
        extra_params_kwargs = {}
        for param_name in self.extra_params:
            if hasattr(p, param_name) and param_name in inspect.signature(self.func).parameters:
                extra_params_kwargs[param_name] = getattr(p, param_name)
        if 'eta' in inspect.signature(self.func).parameters:
            if self.eta != 1.0:
                p.extra_generation_params["Sampler Eta"] = self.eta
            extra_params_kwargs['eta'] = self.eta
        return extra_params_kwargs

    def get_sigmas(self, p, steps): # pylint: disable=unused-argument
        discard_next_to_last_sigma = shared.opts.data.get('schedulers_discard_penultimate', True) if self.config.options.get('discard_next_to_last_sigma', None) is not None else False
        steps += 1 if discard_next_to_last_sigma else 0
        if self.config.options.get('scheduler', None) == 'default' or self.config.options.get('scheduler', None) is None:
            sigmas = self.model_wrap.get_sigmas(steps)
        elif self.config.options.get('scheduler', None) == 'karras':
            sigma_min = p.s_min if p.s_min > 0 else self.model_wrap.sigmas[0].item()
            sigma_max = p.s_max if p.s_max > 0 else self.model_wrap.sigmas[-1].item()
            sigmas = k_sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=shared.device)
        elif self.config.options.get('scheduler', None) == 'exponential':
            sigma_min = p.s_min if p.s_min > 0 else self.model_wrap.sigmas[0].item()
            sigma_max = p.s_max if p.s_max > 0 else self.model_wrap.sigmas[-1].item()
            sigmas = k_sampling.get_sigmas_exponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=shared.device)
        elif self.config.options.get('scheduler', None) == 'polyexponential':
            sigma_min = p.s_min if p.s_min > 0 else self.model_wrap.sigmas[0].item()
            sigma_max = p.s_max if p.s_max > 0 else self.model_wrap.sigmas[-1].item()
            sigmas = k_sampling.get_sigmas_polyexponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=shared.device)
        elif self.config.options.get('scheduler', None) == 'vp':
            sigmas = k_sampling.get_sigmas_vp(n=steps, device=shared.device)
        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def create_noise_sampler(self, x, sigmas, p):
        """For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes"""
        if shared.opts.no_dpmpp_sde_batch_determinism:
            return None
        positive_sigmas = sigmas[sigmas > 0]
        if positive_sigmas.numel() > 0:
            sigma_min = positive_sigmas.min(dim=0)[0]
        else:
            sigma_min = 0
        sigma_max = sigmas.max()
        current_iter_seeds = p.all_seeds[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
        return k_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=current_iter_seeds)

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
        sigmas = self.get_sigmas(p, steps)
        sigma_sched = sigmas[steps - t_enc - 1:]
        xi = x + noise * sigma_sched[0]
        if shared.opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = shared.opts.img2img_extra_noise
            extra_noise_params = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(extra_noise_params)
            noise = extra_noise_params.noise
            xi += noise * shared.opts.img2img_extra_noise

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters
        if 'sigma_min' in parameters:
            ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
            extra_params_kwargs['sigma_min'] = sigma_sched[-2]
        if 'sigma_max' in parameters:
            extra_params_kwargs['sigma_max'] = sigma_sched[0]
        if 'n' in parameters:
            extra_params_kwargs['n'] = len(sigma_sched) - 1
        if 'sigma_sched' in parameters:
            extra_params_kwargs['sigma_sched'] = sigma_sched
        if 'sigmas' in parameters:
            extra_params_kwargs['sigmas'] = sigma_sched
        if self.config.options.get('brownian_noise', False) and 'noise_sampler' in parameters:
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler
        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }
        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, extra_args=extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))
        samples = samples.type(devices.dtype)
        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps = steps or p.steps
        sigmas = self.get_sigmas(p, steps)
        x = x * sigmas[0]
        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters
        if 'sigma_min' in parameters:
            extra_params_kwargs['sigma_min'] = self.model_wrap.sigmas[0].item()
            extra_params_kwargs['sigma_max'] = self.model_wrap.sigmas[-1].item()
            if 'n' in parameters:
                extra_params_kwargs['n'] = steps
        else:
            extra_params_kwargs['sigmas'] = sigmas
        if self.config.options.get('brownian_noise', False) and 'noise_sampler' in parameters:
            noise_sampler = self.create_noise_sampler(x, sigmas, p)
            extra_params_kwargs['noise_sampler'] = noise_sampler
        self.last_latent = x
        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args={
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }, disable=False, callback=self.callback_state, **extra_params_kwargs))
        return samples
