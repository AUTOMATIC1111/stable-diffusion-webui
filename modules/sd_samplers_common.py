from collections import namedtuple
import numpy as np
import torch
from PIL import Image
from modules import devices, images, sd_vae_approx, sd_samplers, sd_vae_taesd, shared
from modules.shared import opts, state

SamplerData = namedtuple('SamplerData', ['name', 'constructor', 'aliases', 'options'])


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
    '''latents -> images [-1, 1]'''
    if approximation is None:
        approximation = approximation_indexes.get(opts.show_progress_type, 0)

    if approximation == 2:
        x_sample = sd_vae_approx.cheap_approximation(sample)
    elif approximation == 1:
        x_sample = sd_vae_approx.model()(sample.to(devices.device, devices.dtype)).detach()
    elif approximation == 3:
        x_sample = sample * 1.5
        x_sample = sd_vae_taesd.decoder_model()(x_sample.to(devices.device, devices.dtype)).detach()
        x_sample = x_sample * 2 - 1
    else:
        if model is None:
            model = shared.sd_model
        x_sample = model.decode_first_stage(sample)

    return x_sample


def single_sample_to_image(sample, approximation=None):
    x_sample = samples_to_images_tensor(sample.unsqueeze(0), approximation)[0] * 0.5 + 0.5

    x_sample = torch.clamp(x_sample, min=0.0, max=1.0)
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)

    return Image.fromarray(x_sample)


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
        x_latent = sd_vae_taesd.encoder_model()(image) / 1.5
    else:
        if model is None:
            model = shared.sd_model
        image = image.to(shared.device, dtype=devices.dtype_vae)
        image = image * 2 - 1
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
