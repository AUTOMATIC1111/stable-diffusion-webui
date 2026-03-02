from collections import namedtuple

import numpy as np
from tqdm import trange

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, sd_samplers_common

import torch
import k_diffusion as K

# Debugging notes - the original method apply_model is being called for sd1.5 is in modules.sd_hijack_utils and is ldm.models.diffusion.ddpm.LatentDiffusion
# For sdxl - OpenAIWrapper will be called, which will call the underlying diffusion_model
# When controlnet is enabled, the underlying model is not available to use, therefore we skip

@torch.no_grad()
def find_noise_for_image(p, cond, uncond, cfg_scale, steps, skip_sdxl_vector):
    x = p.init_latent.clone()

    s_in = x.new_ones([x.shape[0]])
    if shared.sd_model.parameterization == "v":
        dnw = K.external.CompVisVDenoiser(shared.sd_model)
        skip = 1
    else:
        dnw = K.external.CompVisDenoiser(shared.sd_model)
        skip = 0
    sigmas = dnw.get_sigmas(steps).flip(0)

    shared.state.sampling_steps = steps

    for i in trange(1, len(sigmas)):
        shared.state.sampling_step += 1

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigmas[i] * s_in] * 2)

        if shared.sd_model.is_sdxl:
            cond_in = {"crossattn": [torch.cat([uncond['crossattn'], cond['crossattn']])], "vector": [torch.cat([uncond['vector'], cond['vector']])]}
        else:
            cond_in = {"c_concat": [torch.cat([p.image_conditioning] * 2)], "c_crossattn": [torch.cat([uncond, cond])]}

        t = dnw.sigma_to_t(sigma_in)
        dt = sigmas[i] - sigmas[i - 1]
        x += noise_from_model(x, t, dt, sigma_in, cond_in, cfg_scale, dnw, skip, skip_sdxl_vector)

        sd_samplers_common.store_latent(x)

        # This shouldn't be necessary, but solved some VRAM issues
        del x_in, sigma_in, cond_in, t, dt

    shared.state.nextjob()

    return x, sigmas[-1]


# Based on changes suggested by briansemrau in https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/736
@torch.no_grad()
def find_noise_for_image_sigma_adjustment(p, cond, uncond, cfg_scale, steps, correction_factor, skip_sdxl_vector):
    x = p.init_latent.clone()

    s_in = x.new_ones([x.shape[0]])
    if shared.sd_model.parameterization == "v":
        dnw = K.external.CompVisVDenoiser(shared.sd_model)
        skip = 1
    else:
        dnw = K.external.CompVisDenoiser(shared.sd_model)
        skip = 0
    sigmas = dnw.get_sigmas(steps).flip(0)

    shared.state.sampling_steps = steps

    for i in trange(1, len(sigmas)):
        shared.state.sampling_step += 1
        sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
        if shared.sd_model.is_sdxl:
            cond_in = {"crossattn": [torch.cat([uncond['crossattn'], cond['crossattn']])], "vector": [torch.cat([uncond['vector'], cond['vector']])]}
        else:
            cond_in = {"c_concat": [torch.cat([p.image_conditioning] * 2)], "c_crossattn": [torch.cat([uncond, cond])]}

        if i == 1:
            t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
            dt = (sigmas[i] - sigmas[i - 1]) / (2 * sigmas[i])
        else:
            t = dnw.sigma_to_t(sigma_in)
            dt = (sigmas[i] - sigmas[i - 1]) / sigmas[i - 1]

        noise = noise_from_model(x, t, dt, sigma_in, cond_in, cfg_scale, dnw, skip, skip_sdxl_vector)

        if correction_factor > 0: # runs model with previously calculated noise
            recalculated_noise = noise_from_model(x + noise, t, dt, sigma_in, cond_in, cfg_scale, dnw, skip, skip_sdxl_vector)
            noise = recalculated_noise * correction_factor + noise * (1 - correction_factor)

        x += noise

        sd_samplers_common.store_latent(x)

    shared.state.nextjob()

    return x, sigmas[-1]

@torch.no_grad()
def noise_from_model(x, t, dt, sigma_in, cond_in, cfg_scale, dnw, skip, skip_sdxl_vector):

    if cfg_scale == 1:  # Case where denoised_uncond should not be calculated - 50% speedup, also good for sdxl in experiments
        x_in = x
        sigma_in = sigma_in[1:2]
        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)[skip:]]
        cond_in = {k:[v[0][1:2]] for k, v in cond_in.items()}
        if shared.sd_model.is_sdxl:
            num_classes_hack = shared.sd_model.model.diffusion_model.num_classes
            if skip_sdxl_vector:
                shared.sd_model.model.diffusion_model.num_classes = None
                cond_in["vector"][0] = None
            try:
                eps = shared.sd_model.model(x_in * c_in, t[1:2], {"crossattn": cond_in["crossattn"][0], "vector": cond_in["vector"][0]})
            finally:
                shared.sd_model.model.diffusion_model.num_classes = num_classes_hack
        else:
            eps = shared.sd_model.apply_model(x_in * c_in, t[1:2], cond=cond_in)

        return -eps * c_out* dt
    else :
        x_in = torch.cat([x] * 2)

        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)[skip:]]

        if shared.sd_model.is_sdxl:
            num_classes_hack = shared.sd_model.model.diffusion_model.num_classes
            if skip_sdxl_vector:
                shared.sd_model.model.diffusion_model.num_classes = None
                cond_in["vector"][0] = None
            try:
                eps = shared.sd_model.model(x_in * c_in, t, {"crossattn": cond_in["crossattn"][0], "vector": cond_in["vector"][0]} )
            finally:
                shared.sd_model.model.diffusion_model.num_classes = num_classes_hack
        else:
            eps = shared.sd_model.apply_model(x_in * c_in, t, cond=cond_in)

        denoised_uncond, denoised_cond = (eps * c_out).chunk(2)

        denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale

        return -denoised * dt

Cached = namedtuple("Cached", ["noise", "cfg_scale", "steps", "latent", "original_prompt", "original_negative_prompt", "sigma_adjustment", "second_order_correction", "skip_sdxl_vector"])


class Script(scripts.Script):
    def __init__(self):
        self.cache = None

    def title(self):
        return "img2img alternative test"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        info = gr.Markdown('''
        * `CFG Scale` should be 2 or lower.
        ''')

        override_sampler = gr.Checkbox(label="Override `Sampling method` to Euler?(this method is built for it)", value=False, elem_id=self.elem_id("override_sampler"))

        override_prompt = gr.Checkbox(label="Override `prompt` to the same value as `original prompt`?(and `negative prompt`)", value=False, elem_id=self.elem_id("override_prompt"))
        original_prompt = gr.Textbox(label="Original prompt", lines=1, elem_id=self.elem_id("original_prompt"))
        original_negative_prompt = gr.Textbox(label="Original negative prompt", lines=1, elem_id=self.elem_id("original_negative_prompt"))

        override_steps = gr.Checkbox(label="Override `Sampling Steps` to the same value as `Decode steps`?", value=True, elem_id=self.elem_id("override_steps"))
        st = gr.Slider(label="Decode steps", minimum=1, maximum=150, step=1, value=20, elem_id=self.elem_id("st"))

        override_strength = gr.Checkbox(label="Override `Denoising strength` to 1?", value=True, elem_id=self.elem_id("override_strength"))

        cfg = gr.Slider(label="Decode CFG scale", minimum=0.0, maximum=15.0, step=0.1, value=1.0, elem_id=self.elem_id("cfg"))
        randomness = gr.Slider(label="Randomness", minimum=0.0, maximum=1.0, step=0.01, value=0.0, elem_id=self.elem_id("randomness"))
        sigma_adjustment = gr.Checkbox(label="Sigma adjustment for finding noise for image", value=True, elem_id=self.elem_id("sigma_adjustment"))
        second_order_correction = gr.Slider(label="Correct noise by running model again", minimum=0.0, maximum=1.0, step=0.01, value=0.5, elem_id=self.elem_id("second_order_correction"),
                                            info="use 0 (disabled) for original script behaviour, 0.5 reccomended value. Runs the model again to recalculate noise and correct it by given factor. Higher adheres to original image more.")
        noise_sigma_intensity = gr.Slider(label="Weight scaling std vs sigma based", minimum=-1.0, maximum=2.0, step=0.01, value=0.5, elem_id=self.elem_id("noise_sigma_intensity"),
                                          info="use 1 for original script behaviour, 0.5 reccomended value. Decides whether to use fixed sigma value or dynamic standard deviation to scale noise. Lower gives softer images.")
        skip_sdxl_vector = gr.Checkbox(label="Skip sdxl vectors", info="may cause distortion if false", value=True, elem_id=self.elem_id("skip_sdxl_vector"))

        return [
            info,
            override_sampler,
            override_prompt, original_prompt, original_negative_prompt,
            override_steps, st,
            override_strength,
            cfg, randomness, sigma_adjustment, second_order_correction,
            noise_sigma_intensity, skip_sdxl_vector
        ]

    @torch.no_grad()
    def run(self, p, _, override_sampler, override_prompt, original_prompt, original_negative_prompt, override_steps, st, override_strength, cfg, randomness, sigma_adjustment, second_order_correction, noise_sigma_intensity, skip_sdxl_vector):
        # Override
        if override_sampler:
            p.sampler_name = "Euler"
        if override_prompt:
            p.prompt = original_prompt
            p.negative_prompt = original_negative_prompt
        if override_steps:
            p.steps = st
        if override_strength:
            p.denoising_strength = 1.0

        def sample_extra(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
            lat = (p.init_latent.cpu().numpy() * 10).astype(int)

            same_params = self.cache is not None and self.cache.cfg_scale == cfg and self.cache.steps == st \
                                and self.cache.original_prompt == original_prompt \
                                and self.cache.original_negative_prompt == original_negative_prompt \
                                and self.cache.sigma_adjustment == sigma_adjustment \
                                and self.cache.second_order_correction == second_order_correction \
                                and self.cache.skip_sdxl_vector == skip_sdxl_vector

            same_everything = same_params and self.cache.latent.shape == lat.shape and np.abs(self.cache.latent-lat).sum() < 100

            rand_noise = processing.create_random_tensors(p.init_latent.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w, p=p)

            if same_everything:
                rec_noise, sigma_val = self.cache.noise
            else:
                # This prevents a crash, because I don't know how to access the underlying .diffusion_model yet when controlnet is enabled.
                # modules.sd_unet -> we're good
                # scripts.hook -> we're cooked
                if "scripts.hook" in str(shared.sd_model.model.diffusion_model.forward.__module__):
                    print("turn off any controlnets, do 1 pass and then turn controlnet back on to cache noise")
                    p.steps = 1
                    return sd_samplers.create_sampler(p.sampler_name, p.sd_model).sample_img2img(p, p.init_latent, rand_noise, conditioning, unconditional_conditioning, image_conditioning=p.image_conditioning)

                shared.state.job_count += 1
                cond = p.sd_model.get_learned_conditioning(p.batch_size * [original_prompt])
                uncond = p.sd_model.get_learned_conditioning(p.batch_size * [original_negative_prompt])
                if sigma_adjustment:
                    rec_noise, sigma_val = find_noise_for_image_sigma_adjustment(p, cond, uncond, cfg, st, second_order_correction, skip_sdxl_vector)
                else:
                    rec_noise, sigma_val = find_noise_for_image(p, cond, uncond, cfg, st, skip_sdxl_vector)
                self.cache = Cached((rec_noise, sigma_val), cfg, st, lat, original_prompt, original_negative_prompt, sigma_adjustment, second_order_correction, skip_sdxl_vector)

            rec_noise = rec_noise / (rec_noise.std()*(1 - noise_sigma_intensity) + sigma_val*noise_sigma_intensity)

            combined_noise = ((1 - randomness) * rec_noise + randomness * rand_noise) / ((randomness**2 + (1-randomness)**2) ** 0.5)

            sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)

            p.seed = p.seed + 1

            sigmas = sampler.model_wrap.get_sigmas(p.steps)

            noise_dt = combined_noise - (p.init_latent / sigmas[0])

            return sampler.sample_img2img(p, p.init_latent, noise_dt, conditioning, unconditional_conditioning, image_conditioning=p.image_conditioning)

        p.sample = sample_extra

        p.extra_generation_params["Decode prompt"] = original_prompt
        p.extra_generation_params["Decode negative prompt"] = original_negative_prompt
        p.extra_generation_params["Decode CFG scale"] = cfg
        p.extra_generation_params["Decode steps"] = st
        p.extra_generation_params["Randomness"] = randomness
        p.extra_generation_params["Sigma Adjustment"] = sigma_adjustment

        processed = processing.process_images(p)

        return processed
