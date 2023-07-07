import torch

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import noise_like

@torch.no_grad()
def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                    temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                    unconditional_guidance_scale=1., unconditional_conditioning=None,
                    dynamic_threshold=None):
    b, *_, device = *x.shape, x.device

    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        model_output = self.model.apply_model(x, t, c)
    else:
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        if isinstance(c, dict):
            assert isinstance(unconditional_conditioning, dict)
            c_in = dict()
            for k in c:
                if isinstance(c[k], list):
                    c_in[k] = [torch.cat([
                        unconditional_conditioning[k][i],
                        c[k][i]]) for i in range(len(c[k]))]
                else:
                    c_in[k] = torch.cat([
                            unconditional_conditioning[k],
                            c[k]])
        elif isinstance(c, list):
            c_in = list()
            assert isinstance(unconditional_conditioning, list)
            for i in range(len(c)):
                c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
        else:
            c_in = torch.cat([unconditional_conditioning, c])
        model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    if self.model.parameterization == "v":
        e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
    else:
        e_t = model_output

    if score_corrector is not None:
        assert self.model.parameterization == "eps", 'not implemented'
        e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

    alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
    sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
    sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
    # select parameters corresponding to the currently considered timestep
    alphas[index].__str__() # DML Solution: DDIM Sampling does not work without this 'stringify'.
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    # current prediction for x_0
    if self.model.parameterization != "v":
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    else:
        pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

    if quantize_denoised:
        pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

    if dynamic_threshold is not None:
        raise NotImplementedError()

    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    if noise_dropout > 0.:
        noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    return x_prev, pred_x0

DDIMSampler.p_sample_ddim = p_sample_ddim
