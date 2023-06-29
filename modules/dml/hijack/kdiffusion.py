import torch
from tqdm.auto import tqdm
from k_diffusion import sampling
from modules.shared import device


def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
    noise_sampler = sampling.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    if order not in {2, 3}:
        raise ValueError('order should be 2 or 3')
    forward = t_end > t_start
    if not forward and eta:
        raise ValueError('eta must be 0 for reverse sampling')
    h_init = abs(h_init) * (1 if forward else -1)
    atol = torch.tensor(atol, device=device)
    rtol = torch.tensor(rtol, device=device)
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
        return dpm_solver.dpm_solver_fast(x, dpm_solver.t(torch.tensor(sigma_max, device=device)), dpm_solver.t(torch.tensor(sigma_min, device=device)), n, eta, s_noise, noise_sampler)


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = sampling.DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max, device=device)), dpm_solver.t(torch.tensor(sigma_min, device=device)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    if return_info:
        return x, info
    return x

sampling.DPMSolver.dpm_solver_adaptive = dpm_solver_adaptive
sampling.sample_dpm_fast = sample_dpm_fast
sampling.sample_dpm_adaptive = sample_dpm_adaptive
