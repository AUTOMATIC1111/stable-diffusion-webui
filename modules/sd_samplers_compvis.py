import math
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms

import numpy as np
import torch

from modules.shared import state
from modules import sd_samplers_common, prompt_parser, shared


samplers_data_compvis = [
    sd_samplers_common.SamplerData('DDIM', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.ddim.DDIMSampler, model), [], {}),
    sd_samplers_common.SamplerData('PLMS', lambda model: VanillaStableDiffusionSampler(ldm.models.diffusion.plms.PLMSSampler, model), [], {}),
]


class VanillaStableDiffusionSampler:
    def __init__(self, constructor, sd_model):
        self.sampler = constructor(sd_model)
        self.is_plms = hasattr(self.sampler, 'p_sample_plms')
        self.orig_p_sample_ddim = self.sampler.p_sample_plms if self.is_plms else self.sampler.p_sample_ddim
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.sampler_noises = None
        self.step = 0
        self.stop_at = None
        self.eta = None
        self.config = None
        self.last_latent = None

        self.conditioning_key = sd_model.model.conditioning_key

    def number_of_needed_noises(self, p):
        return 0

    def launch_sampling(self, steps, func):
        state.sampling_steps = steps
        state.sampling_step = 0

        try:
            return func()
        except sd_samplers_common.InterruptedException:
            return self.last_latent

    def p_sample_ddim_hook(self, x_dec, cond, ts, unconditional_conditioning, *args, **kwargs):
        if state.interrupted or state.skipped:
            raise sd_samplers_common.InterruptedException

        if self.stop_at is not None and self.step > self.stop_at:
            raise sd_samplers_common.InterruptedException

        # Have to unwrap the inpainting conditioning here to perform pre-processing
        image_conditioning = None
        if isinstance(cond, dict):
            image_conditioning = cond["c_concat"][0]
            cond = cond["c_crossattn"][0]
            unconditional_conditioning = unconditional_conditioning["c_crossattn"][0]

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

        # Wrap the image conditioning back up since the DDIM code can accept the dict directly.
        # Note that they need to be lists because it just concatenates them later.
        if image_conditioning is not None:
            cond = {"c_concat": [image_conditioning], "c_crossattn": [cond]}
            unconditional_conditioning = {"c_concat": [image_conditioning], "c_crossattn": [unconditional_conditioning]}

        res = self.orig_p_sample_ddim(x_dec, cond, ts, unconditional_conditioning=unconditional_conditioning, *args, **kwargs)

        if self.mask is not None:
            self.last_latent = self.init_latent * self.mask + self.nmask * res[1]
        else:
            self.last_latent = res[1]

        sd_samplers_common.store_latent(self.last_latent)

        self.step += 1
        state.sampling_step = self.step
        shared.total_tqdm.update()

        return res

    def initialize(self, p):
        self.eta = p.eta if p.eta is not None else shared.opts.eta_ddim
        if self.eta != 0.0:
            p.extra_generation_params["Eta DDIM"] = self.eta

        for fieldname in ['p_sample_ddim', 'p_sample_plms']:
            if hasattr(self.sampler, fieldname):
                setattr(self.sampler, fieldname, self.p_sample_ddim_hook)

        self.mask = p.mask if hasattr(p, 'mask') else None
        self.nmask = p.nmask if hasattr(p, 'nmask') else None

    def adjust_steps_if_invalid(self, p, num_steps):
        if (self.config.name == 'DDIM' and p.ddim_discretize == 'uniform') or (self.config.name == 'PLMS'):
            valid_step = 999 / (1000 // num_steps)
            if valid_step == math.floor(valid_step):
                return int(valid_step) + 1
        
        return num_steps

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
        steps = self.adjust_steps_if_invalid(p, steps)
        self.initialize(p)

        self.sampler.make_schedule(ddim_num_steps=steps, ddim_eta=self.eta, ddim_discretize=p.ddim_discretize, verbose=False)
        x1 = self.sampler.stochastic_encode(x, torch.tensor([t_enc] * int(x.shape[0])).to(shared.device), noise=noise)

        self.init_latent = x
        self.last_latent = x
        self.step = 0

        # Wrap the conditioning models with additional image conditioning for inpainting model
        if image_conditioning is not None:
            conditioning = {"c_concat": [image_conditioning], "c_crossattn": [conditioning]}
            unconditional_conditioning = {"c_concat": [image_conditioning], "c_crossattn": [unconditional_conditioning]}

        samples = self.launch_sampling(t_enc + 1, lambda: self.sampler.decode(x1, conditioning, t_enc, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning))

        return samples

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        self.initialize(p)

        self.init_latent = None
        self.last_latent = x
        self.step = 0

        steps = self.adjust_steps_if_invalid(p, steps or p.steps)

        # Wrap the conditioning models with additional image conditioning for inpainting model
        # dummy_for_plms is needed because PLMS code checks the first item in the dict to have the right shape
        if image_conditioning is not None:
            conditioning = {"dummy_for_plms": np.zeros((conditioning.shape[0],)), "c_crossattn": [conditioning], "c_concat": [image_conditioning]}
            unconditional_conditioning = {"c_crossattn": [unconditional_conditioning], "c_concat": [image_conditioning]}

        samples_ddim = self.launch_sampling(steps, lambda: self.sampler.sample(S=steps, conditioning=conditioning, batch_size=int(x.shape[0]), shape=x[0].shape, verbose=False, unconditional_guidance_scale=p.cfg_scale, unconditional_conditioning=unconditional_conditioning, x_T=x, eta=self.eta)[0])

        return samples_ddim
