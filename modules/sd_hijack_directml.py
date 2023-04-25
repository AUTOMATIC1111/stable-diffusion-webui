import torch
from tqdm.auto import tqdm

from modules.shared import device

# k-diffusion
from k_diffusion import sampling

def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
    noise_sampler = sampling.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    if order not in {2, 3}:
        raise ValueError('order should be 2 or 3')
    forward = t_end > t_start
    if not forward and eta:
        raise ValueError('eta must be 0 for reverse sampling')
    h_init = abs(h_init) * (1 if forward else -1)
    atol = torch.tensor(atol).to(device)
    rtol = torch.tensor(rtol).to(device)
    s = t_start
    x_prev = x
    accept = True
    pid = sampling.PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
    info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

    while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
        eps_cache = {}
        t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
        if eta:
            sd, su = sampling.get_ancestral_step(self.sigma(s), self.sigma(t), eta)
            t_ = torch.minimum(t_end, self.t(sd))
            su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
        else:
            t_, su = t, 0.

        eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
        denoised = x - self.sigma(s) * eps

        if order == 2:
            x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
            x_high, eps_cache = self.dpm_solver_2_step(x, s, t_, eps_cache=eps_cache)
        else:
            x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
            x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
        delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
        error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
        accept = pid.propose_step(error)
        if accept:
            x_prev = x_low
            x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
            s = t
            info['n_accept'] += 1
        else:
            info['n_reject'] += 1
        info['nfe'] += order
        info['steps'] += 1

        if self.info_callback is not None:
            self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

    return x, info


@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None, eta=0., s_noise=1., noise_sampler=None):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = sampling.DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        return dpm_solver.dpm_solver_fast(x, dpm_solver.t(torch.tensor(sigma_max).to(device)), dpm_solver.t(torch.tensor(sigma_min).to(device)), n, eta, s_noise, noise_sampler)


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = sampling.DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max).to(device)), dpm_solver.t(torch.tensor(sigma_min).to(device)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    if return_info:
        return x, info
    return x

sampling.DPMSolver.dpm_solver_adaptive = dpm_solver_adaptive
sampling.sample_dpm_fast = sample_dpm_fast
sampling.sample_dpm_adaptive = sample_dpm_adaptive

# stablediffusion
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
    print(alphas[index]) # DML Solution: DDIM Sampling does not work without this print.
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

# torch

Generator_init = torch.Generator.__init__
def Generator_init_fix(self, device = None, *args, **kwargs):
    if device is not None and device.type == 'privateuseone':
        return Generator_init(self, 'cpu', *args, **kwargs) # DML Solution: torch.Generator fallback to cpu.
    else:
        return Generator_init(self, device, *args, **kwargs)
torch.Generator.__init__ = Generator_init_fix
