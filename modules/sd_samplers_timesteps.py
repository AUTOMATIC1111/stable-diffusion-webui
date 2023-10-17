# TODO a1111 compatibility module

import torch
from modules import sd_samplers_common, sd_samplers_timesteps_impl, sd_samplers_compvis
from modules.sd_samplers_cfg_denoiser import CFGDenoiser
import modules.shared as shared


samplers_timesteps = [
    ('DDIM', sd_samplers_timesteps_impl.ddim, ['ddim'], {}),
    ('PLMS', sd_samplers_timesteps_impl.plms, ['plms'], {}),
    ('UniPC', sd_samplers_timesteps_impl.unipc, ['unipc'], {}),
]


samplers_data_timesteps = [
    sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: VanillaStableDiffusionSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_timesteps
]


class CompVisTimestepsDenoiser(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_model = model

    def forward(self, input, timesteps, **kwargs): # pylint: disable=redefined-builtin
        return self.inner_model.apply_model(input, timesteps, **kwargs)


class CompVisTimestepsVDenoiser(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_model = model

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return self.inner_model.sqrt_alphas_cumprod[t.to(torch.int), None, None, None] * v + self.inner_model.sqrt_one_minus_alphas_cumprod[t.to(torch.int), None, None, None] * x_t

    def forward(self, input, timesteps, **kwargs): # pylint: disable=redefined-builtin
        model_output = self.inner_model.apply_model(input, timesteps, **kwargs)
        e_t = self.predict_eps_from_z_and_v(input, timesteps, model_output)
        return e_t


class CFGDenoiserTimesteps(CFGDenoiser):

    def __init__(self, sampler):
        super().__init__(sampler)

        self.alphas = shared.sd_model.alphas_cumprod
        self.mask_before_denoising = True

    def get_pred_x0(self, x_in, x_out, sigma):
        ts = sigma.to(dtype=int)

        a_t = self.alphas[ts][:, None, None, None]
        sqrt_one_minus_at = (1 - a_t).sqrt()

        pred_x0 = (x_in - sqrt_one_minus_at * x_out) / a_t.sqrt()

        return pred_x0

    @property
    def inner_model(self):
        if self.model_wrap is None:
            denoiser = CompVisTimestepsVDenoiser if shared.sd_model.parameterization == "v" else CompVisTimestepsDenoiser
            self.model_wrap = denoiser(shared.sd_model)

        return self.model_wrap


VanillaStableDiffusionSampler = sd_samplers_compvis.VanillaStableDiffusionSampler
