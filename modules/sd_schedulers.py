import dataclasses
import torch
import k_diffusion
import numpy as np

from modules import shared


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma


k_diffusion.sampling.to_d = to_d


@dataclasses.dataclass
class Scheduler:
    name: str
    label: str
    function: any

    default_rho: float = -1
    need_inner_model: bool = False
    aliases: list = None


def uniform(n, sigma_min, sigma_max, inner_model, device):
    return inner_model.get_sigmas(n).to(device)


def sgm_uniform(n, sigma_min, sigma_max, inner_model, device):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))
    sigs = [
        inner_model.t_to_sigma(ts)
        for ts in torch.linspace(start, end, n + 1)[:-1]
    ]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def get_align_your_steps_sigmas(n, sigma_min, sigma_max, device):
    # https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html
    def loglinear_interp(t_steps, num_steps):
        """
        Performs log-linear interpolation of a given array of decreasing numbers.
        """
        xs = np.linspace(0, 1, len(t_steps))
        ys = np.log(t_steps[::-1])

        new_xs = np.linspace(0, 1, num_steps)
        new_ys = np.interp(new_xs, xs, ys)

        interped_ys = np.exp(new_ys)[::-1].copy()
        return interped_ys

    if shared.sd_model.is_sdxl:
        sigmas = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    else:
        # Default to SD 1.5 sigmas.
        sigmas = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

    if n != len(sigmas):
        sigmas = np.append(loglinear_interp(sigmas, n), [0.0])
    else:
        sigmas.append(0.0)

    return torch.FloatTensor(sigmas).to(device)


def kl_optimal(n, sigma_min, sigma_max, device):
    alpha_min = torch.arctan(torch.tensor(sigma_min, device=device))
    alpha_max = torch.arctan(torch.tensor(sigma_max, device=device))
    step_indices = torch.arange(n + 1, device=device)
    sigmas = torch.tan(step_indices / n * alpha_min + (1.0 - step_indices / n) * alpha_max)
    return sigmas

def ddim_cfgpp(n, sigma_min, sigma_max, inner_model, device):
    if hasattr(inner_model, 'alphas_cumprod'):
        # For timestep-based samplers
        alphas_cumprod = inner_model.alphas_cumprod
    elif hasattr(inner_model, 'inner_model'):
        # For k-diffusion samplers
        alphas_cumprod = inner_model.inner_model.alphas_cumprod
    else:
        raise AttributeError("Cannot find alphas_cumprod in the model")

    timesteps = torch.linspace(0, 999, n, device=device).long()
    alphas = alphas_cumprod[timesteps]
    alphas_prev = alphas_cumprod[torch.nn.functional.pad(timesteps[:-1], pad=(1, 0))]
    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
    sigmas = sqrt_one_minus_alphas / torch.sqrt(alphas)
    
    # Ensure sigmas are in descending order
    sigmas = torch.flip(sigmas, [0])
    
    # Add a final sigma of 0 for the last step
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    
    return sigmas.to(device)


schedulers = [
    Scheduler('automatic', 'Automatic', None),
    Scheduler('uniform', 'Uniform', uniform, need_inner_model=True),
    Scheduler('karras', 'Karras', k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler('exponential', 'Exponential', k_diffusion.sampling.get_sigmas_exponential),
    Scheduler('polyexponential', 'Polyexponential', k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler('sgm_uniform', 'SGM Uniform', sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler('kl_optimal', 'KL Optimal', kl_optimal),
    Scheduler('align_your_steps', 'Align Your Steps', get_align_your_steps_sigmas),
    Scheduler('ddim_cfgpp', 'CFG++', ddim_cfgpp, need_inner_model=True),
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
