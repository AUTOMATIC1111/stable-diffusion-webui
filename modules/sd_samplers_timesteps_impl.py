import torch
import tqdm
import k_diffusion.sampling
import numpy as np

from modules import shared
from modules.models.diffusion.uni_pc import uni_pc


@torch.no_grad()
def ddim(model, x, timesteps, extra_args=None, callback=None, disable=None, eta=0.0):
    alphas_cumprod = model.inner_model.inner_model.alphas_cumprod
    alphas = alphas_cumprod[timesteps]
    alphas_prev = alphas_cumprod[torch.nn.functional.pad(timesteps[:-1], pad=(1, 0))].to(torch.float64 if x.device.type != 'mps' and x.device.type != 'xpu' else torch.float32)
    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
    sigmas = eta * np.sqrt((1 - alphas_prev.cpu().numpy()) / (1 - alphas.cpu()) * (1 - alphas.cpu() / alphas_prev.cpu().numpy()))

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones((x.shape[0]))
    s_x = x.new_ones((x.shape[0], 1, 1, 1))
    for i in tqdm.trange(len(timesteps) - 1, disable=disable):
        index = len(timesteps) - 1 - i

        e_t = model(x, timesteps[index].item() * s_in, **extra_args)

        a_t = alphas[index].item() * s_x
        a_prev = alphas_prev[index].item() * s_x
        sigma_t = sigmas[index].item() * s_x
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * k_diffusion.sampling.torch.randn_like(x)
        x = a_prev.sqrt() * pred_x0 + dir_xt + noise

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    return x


@torch.no_grad()
def plms(model, x, timesteps, extra_args=None, callback=None, disable=None):
    alphas_cumprod = model.inner_model.inner_model.alphas_cumprod
    alphas = alphas_cumprod[timesteps]
    alphas_prev = alphas_cumprod[torch.nn.functional.pad(timesteps[:-1], pad=(1, 0))].to(torch.float64 if x.device.type != 'mps' and x.device.type != 'xpu' else torch.float32)
    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    s_x = x.new_ones((x.shape[0], 1, 1, 1))
    old_eps = []

    def get_x_prev_and_pred_x0(e_t, index):
        # select parameters corresponding to the currently considered timestep
        a_t = alphas[index].item() * s_x
        a_prev = alphas_prev[index].item() * s_x
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1. - a_prev).sqrt() * e_t
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        return x_prev, pred_x0

    for i in tqdm.trange(len(timesteps) - 1, disable=disable):
        index = len(timesteps) - 1 - i
        ts = timesteps[index].item() * s_in
        t_next = timesteps[max(index - 1, 0)].item() * s_in

        e_t = model(x, ts, **extra_args)

        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = model(x_prev, t_next, **extra_args)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        else:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        old_eps.append(e_t)
        if len(old_eps) >= 4:
            old_eps.pop(0)

        x = x_prev

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    return x


class UniPCCFG(uni_pc.UniPC):
    def __init__(self, cfg_model, extra_args, callback, *args, **kwargs):
        super().__init__(None, *args, **kwargs)

        def after_update(x, model_x):
            callback({'x': x, 'i': self.index, 'sigma': 0, 'sigma_hat': 0, 'denoised': model_x})
            self.index += 1

        self.cfg_model = cfg_model
        self.extra_args = extra_args
        self.callback = callback
        self.index = 0
        self.after_update = after_update

    def get_model_input_time(self, t_continuous):
        return (t_continuous - 1. / self.noise_schedule.total_N) * 1000.

    def model(self, x, t):
        t_input = self.get_model_input_time(t)

        res = self.cfg_model(x, t_input, **self.extra_args)

        return res


def unipc(model, x, timesteps, extra_args=None, callback=None, disable=None, is_img2img=False):
    alphas_cumprod = model.inner_model.inner_model.alphas_cumprod

    ns = uni_pc.NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
    t_start = timesteps[-1] / 1000 + 1 / 1000 if is_img2img else None  # this is likely off by a bit - if someone wants to fix it please by all means
    unipc_sampler = UniPCCFG(model, extra_args, callback, ns, predict_x0=True, thresholding=False, variant=shared.opts.uni_pc_variant)
    x = unipc_sampler.sample(x, steps=len(timesteps), t_start=t_start, skip_type=shared.opts.uni_pc_skip_type, method="multistep", order=shared.opts.uni_pc_order, lower_order_final=shared.opts.uni_pc_lower_order_final)

    return x
