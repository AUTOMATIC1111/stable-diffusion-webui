import inspect
from collections import namedtuple
import numpy as np
import torch
from PIL import Image
from modules import devices, images, sd_vae_approx, sd_samplers, sd_vae_taesd, shared, sd_models
from modules.shared import opts, state
import k_diffusion.sampling


SamplerDataTuple = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])


class SamplerData(SamplerDataTuple):
    def total_steps(self, steps):
        if self.options.get("second_order", False):
            steps = steps * 2

        return steps


def setup_img2img_steps(p, steps=None):
    if opts.img2img_fix_steps or steps is not None:
        requested_steps = (steps or p.steps)
        steps = int(requested_steps / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = requested_steps - 1
    else:
        steps = p.steps
        t_enc = int(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc


approximation_indexes = {"Full": 0, "Approx NN": 1, "Approx cheap": 2, "TAESD": 3}


def samples_to_images_tensor(sample, approximation=None, model=None):
    """Transforms 4-channel latent space images into 3-channel RGB image tensors, with values in range [-1, 1]."""

    if approximation is None or (shared.state.interrupted and opts.live_preview_fast_interrupt):
        approximation = approximation_indexes.get(opts.show_progress_type, 0)

        from modules import lowvram
        if approximation == 0 and lowvram.is_enabled(shared.sd_model) and not shared.opts.live_preview_allow_lowvram_full:
            approximation = 1

    if approximation == 2:
        x_sample = sd_vae_approx.cheap_approximation(sample)
    elif approximation == 1:
        x_sample = sd_vae_approx.model()(sample.to(devices.device, devices.dtype)).detach()
    elif approximation == 3:
        x_sample = sd_vae_taesd.decoder_model()(sample.to(devices.device, devices.dtype)).detach()
        x_sample = x_sample * 2 - 1
    else:
        if model is None:
            model = shared.sd_model
        with torch.no_grad(), devices.without_autocast(): # fixes an issue with unstable VAEs that are flaky even in fp32
            x_sample = model.decode_first_stage(sample.to(model.first_stage_model.dtype))

    return x_sample


def single_sample_to_image(sample, approximation=None):
    x_sample = samples_to_images_tensor(sample.unsqueeze(0), approximation)[0] * 0.5 + 0.5

    x_sample = torch.clamp(x_sample, min=0.0, max=1.0)
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)

    return Image.fromarray(x_sample)


def decode_first_stage(model, x):
    x = x.to(devices.dtype_vae)
    approx_index = approximation_indexes.get(opts.sd_vae_decode_method, 0)
    return samples_to_images_tensor(x, approx_index, model)


def sample_to_image(samples, index=0, approximation=None):
    return single_sample_to_image(samples[index], approximation)


def samples_to_image_grid(samples, approximation=None):
    return images.image_grid([single_sample_to_image(sample, approximation) for sample in samples])


def images_tensor_to_samples(image, approximation=None, model=None):
    '''image[0, 1] -> latent'''
    if approximation is None:
        approximation = approximation_indexes.get(opts.sd_vae_encode_method, 0)

    if approximation == 3:
        image = image.to(devices.device, devices.dtype)
        x_latent = sd_vae_taesd.encoder_model()(image)
    else:
        if model is None:
            model = shared.sd_model
        model.first_stage_model.to(devices.dtype_vae)

        image = image.to(shared.device, dtype=devices.dtype_vae)
        image = image * 2 - 1
        if len(image) > 1:
            x_latent = torch.stack([
                model.get_first_stage_encoding(
                    model.encode_first_stage(torch.unsqueeze(img, 0))
                )[0]
                for img in image
            ])
        else:
            x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))

    return x_latent


def store_latent(decoded):
    state.current_latent = decoded

    if opts.live_previews_enable and opts.show_progress_every_n_steps > 0 and shared.state.sampling_step % opts.show_progress_every_n_steps == 0:
        if not shared.parallel_processing_allowed:
            shared.state.assign_current_image(sample_to_image(decoded))


def is_sampler_using_eta_noise_seed_delta(p):
    """returns whether sampler from config will use eta noise seed delta for image creation"""

    sampler_config = sd_samplers.find_sampler_config(p.sampler_name)

    eta = p.eta

    if eta is None and p.sampler is not None:
        eta = p.sampler.eta

    if eta is None and sampler_config is not None:
        eta = 0 if sampler_config.options.get("default_eta_is_0", False) else 1.0

    if eta == 0:
        return False

    return sampler_config.options.get("uses_ensd", False)


class InterruptedException(BaseException):
    pass


def replace_torchsde_browinan():
    import torchsde._brownian.brownian_interval

    def torchsde_randn(size, dtype, device, seed):
        return devices.randn_local(seed, size).to(device=device, dtype=dtype)

    torchsde._brownian.brownian_interval._randn = torchsde_randn


replace_torchsde_browinan()


def apply_refiner(cfg_denoiser, sigma=None):
    if opts.refiner_switch_by_sample_steps or sigma is None:
        completed_ratio = cfg_denoiser.step / cfg_denoiser.total_steps
        cfg_denoiser.p.extra_generation_params["Refiner switch by sampling steps"] = True

    else:
        # torch.max(sigma) only to handle rare case where we might have different sigmas in the same batch
        try:
            timestep = torch.argmin(torch.abs(cfg_denoiser.inner_model.sigmas.to(sigma.device) - torch.max(sigma)))
        except AttributeError:  # for samplers that don't use sigmas (DDIM) sigma is actually the timestep
            timestep = torch.max(sigma).to(dtype=int)
        completed_ratio = (999 - timestep) / 1000

    refiner_switch_at = cfg_denoiser.p.refiner_switch_at
    refiner_checkpoint_info = cfg_denoiser.p.refiner_checkpoint_info

    if refiner_switch_at is not None and completed_ratio < refiner_switch_at:
        return False

    if refiner_checkpoint_info is None or shared.sd_model.sd_checkpoint_info == refiner_checkpoint_info:
        return False

    if getattr(cfg_denoiser.p, "enable_hr", False):
        is_second_pass = cfg_denoiser.p.is_hr_pass

        if opts.hires_fix_refiner_pass == "first pass" and is_second_pass:
            return False

        if opts.hires_fix_refiner_pass == "second pass" and not is_second_pass:
            return False

        if opts.hires_fix_refiner_pass != "second pass":
            cfg_denoiser.p.extra_generation_params['Hires refiner'] = opts.hires_fix_refiner_pass

    cfg_denoiser.p.extra_generation_params['Refiner'] = refiner_checkpoint_info.short_title
    cfg_denoiser.p.extra_generation_params['Refiner switch at'] = refiner_switch_at

    with sd_models.SkipWritingToConfig():
        sd_models.reload_model_weights(info=refiner_checkpoint_info)

    devices.torch_gc()
    cfg_denoiser.p.setup_conds()
    cfg_denoiser.update_inner_model()

    return True


class TorchHijack:
    """This is here to replace torch.randn_like of k-diffusion.

    k-diffusion has random_sampler argument for most samplers, but not for all, so
    this is needed to properly replace every use of torch.randn_like.

    We need to replace to make images generated in batches to be same as images generated individually."""

    def __init__(self, p):
        self.rng = p.rng

    def __getattr__(self, item):
        if item == 'randn_like':
            return self.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        return self.rng.next()


class Sampler:
    def __init__(self, funcname):
        self.funcname = funcname
        self.func = funcname
        self.extra_params = []
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config: SamplerData = None  # set by the function calling the constructor
        self.last_latent = None
        self.s_min_uncond = None
        self.s_churn = 0.0
        self.s_tmin = 0.0
        self.s_tmax = float('inf')
        self.s_noise = 1.0

        self.eta_option_field = 'eta_ancestral'
        self.eta_infotext_field = 'Eta'
        self.eta_default = 1.0

        self.conditioning_key = getattr(shared.sd_model.model, 'conditioning_key', 'crossattn')

        self.p = None
        self.model_wrap_cfg = None
        self.sampler_extra_args = None
        self.options = {}

    def callback_state(self, d):
        step = d['i']

        if self.stop_at is not None and step > self.stop_at:
            raise InterruptedException

        state.sampling_step = step
        shared.total_tqdm.update()

    def launch_sampling(self, steps, func):
        self.model_wrap_cfg.steps = steps
        self.model_wrap_cfg.total_steps = self.config.total_steps(steps)
        state.sampling_steps = steps
        state.sampling_step = 0

        try:
            return func()
        except RecursionError:
            print(
                'Encountered RecursionError during sampling, returning last latent. '
                'rho >5 with a polyexponential scheduler may cause this error. '
                'You should try to use a smaller rho value instead.'
            )
            return self.last_latent
        except InterruptedException:
            return self.last_latent

    def number_of_needed_noises(self, p):
        return p.steps

    def initialize(self, p) -> dict:
        self.p = p
        self.model_wrap_cfg.p = p
        self.model_wrap_cfg.mask = p.mask if hasattr(p, 'mask') else None
        self.model_wrap_cfg.nmask = p.nmask if hasattr(p, 'nmask') else None
        self.model_wrap_cfg.step = 0
        self.model_wrap_cfg.image_cfg_scale = getattr(p, 'image_cfg_scale', None)
        self.eta = p.eta if p.eta is not None else getattr(opts, self.eta_option_field, 0.0)
        self.s_min_uncond = getattr(p, 's_min_uncond', 0.0)

        k_diffusion.sampling.torch = TorchHijack(p)

        extra_params_kwargs = {}
        for param_name in self.extra_params:
            if hasattr(p, param_name) and param_name in inspect.signature(self.func).parameters:
                extra_params_kwargs[param_name] = getattr(p, param_name)

        if 'eta' in inspect.signature(self.func).parameters:
            if self.eta != self.eta_default:
                p.extra_generation_params[self.eta_infotext_field] = self.eta

            extra_params_kwargs['eta'] = self.eta

        if len(self.extra_params) > 0:
            s_churn = getattr(opts, 's_churn', p.s_churn)
            s_tmin = getattr(opts, 's_tmin', p.s_tmin)
            s_tmax = getattr(opts, 's_tmax', p.s_tmax) or self.s_tmax # 0 = inf
            s_noise = getattr(opts, 's_noise', p.s_noise)

            if 's_churn' in extra_params_kwargs and s_churn != self.s_churn:
                extra_params_kwargs['s_churn'] = s_churn
                p.s_churn = s_churn
                p.extra_generation_params['Sigma churn'] = s_churn
            if 's_tmin' in extra_params_kwargs and s_tmin != self.s_tmin:
                extra_params_kwargs['s_tmin'] = s_tmin
                p.s_tmin = s_tmin
                p.extra_generation_params['Sigma tmin'] = s_tmin
            if 's_tmax' in extra_params_kwargs and s_tmax != self.s_tmax:
                extra_params_kwargs['s_tmax'] = s_tmax
                p.s_tmax = s_tmax
                p.extra_generation_params['Sigma tmax'] = s_tmax
            if 's_noise' in extra_params_kwargs and s_noise != self.s_noise:
                extra_params_kwargs['s_noise'] = s_noise
                p.s_noise = s_noise
                p.extra_generation_params['Sigma noise'] = s_noise

        return extra_params_kwargs

    def create_noise_sampler(self, x, sigmas, p):
        """For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes"""
        if shared.opts.no_dpmpp_sde_batch_determinism:
            return None

        from k_diffusion.sampling import BrownianTreeNoiseSampler
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        current_iter_seeds = p.all_seeds[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
        return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=current_iter_seeds)

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        raise NotImplementedError()

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        raise NotImplementedError()

    def add_infotext(self, p):
        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        if self.model_wrap_cfg.padded_cond_uncond_v0:
            p.extra_generation_params["Pad conds v0"] = True
