from collections import namedtuple

import numpy as np
from tqdm import trange

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, prompt_parser
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state

import torch
import k_diffusion as K

from PIL import Image
from torch import autocast
from einops import rearrange, repeat


def find_noise_for_image(p, cond, uncond, cfg_scale, steps):
    x = p.init_latent

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(shared.sd_model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    shared.state.sampling_steps = steps

    for i in trange(1, len(sigmas)):
        shared.state.sampling_step += 1

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigmas[i] * s_in] * 2)
        cond_in = torch.cat([uncond, cond])

        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]
        t = dnw.sigma_to_t(sigma_in)

        eps = shared.sd_model.apply_model(x_in * c_in, t, cond=cond_in)
        denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

        denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale

        d = (x - denoised) / sigmas[i]
        dt = sigmas[i] - sigmas[i - 1]

        x = x + d * dt

        sd_samplers.store_latent(x)

        # This shouldn't be necessary, but solved some VRAM issues
        del x_in, sigma_in, cond_in, c_out, c_in, t,
        del eps, denoised_uncond, denoised_cond, denoised, d, dt

    shared.state.nextjob()

    return x / x.std()


Cached = namedtuple("Cached", ["noise", "cfg_scale", "steps", "latent", "original_prompt"])


class Script(scripts.Script):
    def __init__(self):
        self.cache = None

    def title(self):
        return "img2img alternative test"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        original_prompt = gr.Textbox(label="Original prompt", lines=1)
        cfg = gr.Slider(label="Decode CFG scale", minimum=0.0, maximum=15.0, step=0.1, value=1.0)
        st = gr.Slider(label="Decode steps", minimum=1, maximum=150, step=1, value=50)

        return [original_prompt, cfg, st]

    def run(self, p, original_prompt, cfg, st):
        p.batch_size = 1
        p.batch_count = 1

        def sample_extra(x, conditioning, unconditional_conditioning):
            lat = (p.init_latent.cpu().numpy() * 10).astype(int)

            same_params = self.cache is not None and self.cache.cfg_scale == cfg and self.cache.steps == st and self.cache.original_prompt == original_prompt
            same_everything = same_params and self.cache.latent.shape == lat.shape and np.abs(self.cache.latent-lat).sum() < 100

            if same_everything:
                noise = self.cache.noise
            else:
                shared.state.job_count += 1
                cond = p.sd_model.get_learned_conditioning(p.batch_size * [original_prompt])
                uncond = p.sd_model.get_learned_conditioning(p.batch_size * [""])
                noise = find_noise_for_image(p, cond, uncond, cfg, st)
                self.cache = Cached(noise, cfg, st, lat, original_prompt)

            sampler = samplers[p.sampler_index].constructor(p.sd_model)

            samples_ddim = sampler.sample(p, noise, conditioning, unconditional_conditioning)
            return samples_ddim

        p.sample = sample_extra

        p.extra_generation_params = {
            "Decode prompt": original_prompt,
            "Decode CFG scale": cfg,
            "Decode steps": st,
        }

        processed = processing.process_images(p)

        return processed

