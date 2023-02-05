from collections import namedtuple
import numpy as np
import torch
from PIL import Image
from modules import devices, processing, images, sd_vae_approx

from modules.shared import opts, state
import modules.shared as shared

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


approximation_indexes = {"Full": 0, "Approx NN": 1, "Approx cheap": 2}


def single_sample_to_image(sample, approximation=None):
    if approximation is None:
        approximation = approximation_indexes.get(opts.show_progress_type, 0)

    if approximation == 2:
        x_sample = sd_vae_approx.cheap_approximation(sample)
    elif approximation == 1:
        x_sample = sd_vae_approx.model()(sample.to(devices.device, devices.dtype).unsqueeze(0))[0].detach()
    else:
        x_sample = processing.decode_first_stage(shared.sd_model, sample.unsqueeze(0))[0]

    x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)
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


class InterruptedException(BaseException):
    pass
