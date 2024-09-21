import dataclasses
import torch
import k_diffusion
import numpy as np
from scipy import stats

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


def simple_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = len(inner_model.sigmas) / n
    for x in range(n):
        sigs += [float(inner_model.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def normal_scheduler(n, sigma_min, sigma_max, inner_model, device, sgm=False, floor=False):
    start = inner_model.sigma_to_t(torch.tensor(sigma_max))
    end = inner_model.sigma_to_t(torch.tensor(sigma_min))

    if sgm:
        timesteps = torch.linspace(start, end, n + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, n)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(inner_model.t_to_sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def ddim_scheduler(n, sigma_min, sigma_max, inner_model, device):
    sigs = []
    ss = max(len(inner_model.sigmas) // n, 1)
    x = 1
    while x < len(inner_model.sigmas):
        sigs += [float(inner_model.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs).to(device)


def karras_exponential_scheduler(n, sigma_min, sigma_max, device, blend_factor=0.3, sharpen_factor=0.9):
    # Optional: Adjust sigma_max to fine-tune the range of noise levels
    # Example: Increase by 10%; modify the multiplier as needed (e.g., 1.1 for 10% increase)
    sigma_max = sigma_max * 1.1  # Adjust this multiplier as needed, e.g., 1.1 for a 10% increase
      # Initialize sigmas to None to avoid UnboundLocalError in case of failure during assignment
    sigmas_karras, sigmas_exponential = None, None
    try:
         # Generate sigma schedules using Karras and Exponential methods
        # These functions are from the k_diffusion module and are crucial for generating the noise schedule
        sigmas_karras = k_diffusion.sampling.get_sigmas_karras(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
        sigmas_exponential = k_diffusion.sampling.get_sigmas_exponential(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)

        # Print the lengths of the generated sequences for debugging purposes
        #print(f"Length before resampling: Karras - {len(sigmas_karras)}, Exponential - {len(sigmas_exponential)}")

        # Check if lengths are different; resample if necessary to match lengths
        if len(sigmas_karras) != len(sigmas_exponential):
            # Resample both sigmas to match the longer sequence length; ensures consistent blending
            max_length = max(len(sigmas_karras), len(sigmas_exponential))
            sigmas_karras = resample_sigmas(sigmas_karras, max_length, device)
            sigmas_exponential = resample_sigmas(sigmas_exponential, max_length, device)

    except Exception as e:
        # Handle errors during sigma generation; assign fallback empty tensors if an error occurs
        print(f"Error generating sigmas: {e}")
        sigmas_karras = torch.zeros(n).to(device)
        sigmas_exponential = torch.zeros(n).to(device)

    # Ensure sigmas have been assigned correctly; raise an error if not
    if sigmas_karras is None or sigmas_exponential is None:
        raise ValueError("Failed to generate or assign sigmas correctly.")

    # Create a linear tensor from 0 to 1 to represent progress over the length of sigmas
    progress = torch.linspace(0, 1, len(sigmas_karras)).to(device)
    # Calculate a dynamic blend factor that decreases from blend_factor to 0
    dynamic_blend_factor = (1 - progress) * blend_factor
    # Blend the Karras and Exponential sigmas based on the dynamic blend factor
    sigs = (sigmas_karras * (1 - dynamic_blend_factor) + sigmas_exponential * dynamic_blend_factor)

   # Trim the blended sigmas if they exceed the required number of steps
    if len(sigs) > n:
        sigs = sigs[:n]

    # Apply sharpening to sigmas below a certain threshold to enhance sharpness
    # Modify sharpen_factor to adjust sharpening intensity
    sharpen_mask = torch.where(sigs < sigma_min * 1.5, sharpen_factor, 1.0).to(device)
    sigs = sigs * sharpen_mask

    # Return the final blended and adjusted sigmas on the specified device
    return sigs.to(device)
    

def beta_scheduler(n, sigma_min, sigma_max, inner_model, device):
    # From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024) """
    alpha = shared.opts.beta_dist_alpha
    beta = shared.opts.beta_dist_beta
    timesteps = 1 - np.linspace(0, 1, n)
    timesteps = [stats.beta.ppf(x, alpha, beta) for x in timesteps]
    sigmas = [sigma_min + (x * (sigma_max-sigma_min)) for x in timesteps]
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)


schedulers = [
    Scheduler('automatic', 'Automatic', None),
    Scheduler('uniform', 'Uniform', uniform, need_inner_model=True),
    Scheduler('karras', 'Karras', k_diffusion.sampling.get_sigmas_karras, default_rho=7.0),
    Scheduler('exponential', 'Exponential', k_diffusion.sampling.get_sigmas_exponential),
    Scheduler('polyexponential', 'Polyexponential', k_diffusion.sampling.get_sigmas_polyexponential, default_rho=1.0),
    Scheduler('sgm_uniform', 'SGM Uniform', sgm_uniform, need_inner_model=True, aliases=["SGMUniform"]),
    Scheduler('kl_optimal', 'KL Optimal', kl_optimal),
    Scheduler('align_your_steps', 'Align Your Steps', get_align_your_steps_sigmas),
    Scheduler('simple', 'Simple', simple_scheduler, need_inner_model=True),
    Scheduler('normal', 'Normal', normal_scheduler, need_inner_model=True),
    Scheduler('ddim', 'DDIM', ddim_scheduler, need_inner_model=True),
    Scheduler('beta', 'Beta', beta_scheduler, need_inner_model=True),
    Scheduler('karras_exponential', 'Karras Exponential', karras_exponential_scheduler),
   
]

schedulers_map = {**{x.name: x for x in schedulers}, **{x.label: x for x in schedulers}}
