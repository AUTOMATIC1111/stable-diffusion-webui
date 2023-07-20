from collections import namedtuple
import numpy as np
import torch
from PIL import Image
from modules import devices, processing, images, sd_vae_approx, sd_samplers, sd_vae_taesd
from modules.shared import opts, state
import modules.shared as shared


SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])
approximation_indexes = {"Full VAE": 0, "Approximate NN": 1, "Approximate simple": 2, "TAESD": 3}


def setup_img2img_steps(p, steps=None):
    if opts.img2img_fix_steps or steps is not None:
        requested_steps = (steps or p.steps)
        steps = int(requested_steps / min(p.denoising_strength, 0.999)) if p.denoising_strength > 0 else 0
        t_enc = requested_steps - 1
    else:
        steps = p.steps
        t_enc = int(min(p.denoising_strength, 0.999) * steps)

    return steps, t_enc


def single_sample_to_image(sample, approximation=None):
    if approximation is None:
        approximation = approximation_indexes.get(opts.show_progress_type, 0)
    if approximation == 0:
        x_sample = processing.decode_first_stage(shared.sd_model, sample.unsqueeze(0))[0] * 0.5 + 0.5
    elif approximation == 1:
        x_sample = sd_vae_approx.model()(sample.to(devices.device, devices.dtype).unsqueeze(0))[0].detach() * 0.5 + 0.5
    elif approximation == 2:
        x_sample = sd_vae_approx.cheap_approximation(sample) * 0.5 + 0.5
    elif approximation == 3:
        x_sample = sample * 1.5
        x_sample = sd_vae_taesd.model()(x_sample.to(devices.device, devices.dtype).unsqueeze(0))[0].detach()
    else:
        shared.log.warning(f"Unknown image decode type: {approximation}")
        return Image.new(mode="RGB", size=(512, 512))
    x_sample = torch.clamp(255 * x_sample, min=0.0, max=255).cpu()
    x_sample = np.moveaxis(x_sample.numpy(), 0, 2).astype(np.uint8)
    return Image.fromarray(x_sample)


def sample_to_image(samples, index=0, approximation=None):
    return single_sample_to_image(samples[index], approximation)


def samples_to_image_grid(samples, approximation=None):
    return images.image_grid([single_sample_to_image(sample, approximation) for sample in samples])


def store_latent(decoded):
    state.current_latent = decoded
    if opts.live_previews_enable and opts.show_progress_every_n_steps > 0 and shared.state.sampling_step % opts.show_progress_every_n_steps == 0:
        if not shared.parallel_processing_allowed:
            shared.state.assign_current_image(sample_to_image(decoded))


def is_sampler_using_eta_noise_seed_delta(p):
    """returns whether sampler from config will use eta noise seed delta for image creation"""
    sampler_config = sd_samplers.find_sampler_config(p.sampler_name)
    eta = p.eta
    if not hasattr(p.sampler, "eta"):
        return False
    if eta is None and p.sampler is not None:
        eta = p.sampler.eta
    if eta is None and sampler_config is not None:
        eta = 0 if sampler_config.options.get("default_eta_is_0", False) else 1.0
    if eta == 0:
        return False
    return sampler_config.options.get("uses_ensd", False)


class InterruptedException(BaseException):
    pass
