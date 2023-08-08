import sys
import contextlib
from functools import lru_cache

import torch
from modules import errors, rng_philox

if sys.platform == "darwin":
    from modules import mac_specific


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        return mac_specific.has_mps


def get_cuda_device_string():
    from modules import shared

    if shared.cmd_opts.device_id is not None:
        return f"cuda:{shared.cmd_opts.device_id}"

    return "cuda"


def get_optimal_device_name():
    if torch.cuda.is_available():
        return get_cuda_device_string()

    if has_mps():
        return "mps"

    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task):
    from modules import shared

    if task in shared.cmd_opts.use_cpu:
        return cpu

    return get_optimal_device()


def torch_gc():

    if torch.cuda.is_available():
        with torch.cuda.device(get_cuda_device_string()):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    if has_mps():
        mac_specific.torch_mps_gc()


def enable_tf32():
    if torch.cuda.is_available():

        # enabling benchmark option seems to enable a range of cards to do fp16 when they otherwise can't
        # see https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4407
        if any(torch.cuda.get_device_capability(devid) == (7, 5) for devid in range(0, torch.cuda.device_count())):
            torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")

cpu: torch.device = torch.device("cpu")
device: torch.device = None
device_interrogate: torch.device = None
device_gfpgan: torch.device = None
device_esrgan: torch.device = None
device_codeformer: torch.device = None
dtype: torch.dtype = torch.float16
dtype_vae: torch.dtype = torch.float16
dtype_unet: torch.dtype = torch.float16
unet_needs_upcast = False


def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input


def cond_cast_float(input):
    return input.float() if unet_needs_upcast else input


nv_rng = None


def randn(seed, shape):
    """Generate a tensor with random numbers from a normal distribution using seed.

    Uses the seed parameter to set the global torch seed; to generate more with that seed, use randn_like/randn_without_seed."""

    from modules.shared import opts

    manual_seed(seed)

    if opts.randn_source == "NV":
        return torch.asarray(nv_rng.randn(shape), device=device)

    if opts.randn_source == "CPU" or device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)

    return torch.randn(shape, device=device)


def randn_local(seed, shape):
    """Generate a tensor with random numbers from a normal distribution using seed.

    Does not change the global random number generator. You can only generate the seed's first tensor using this function."""

    from modules.shared import opts

    if opts.randn_source == "NV":
        rng = rng_philox.Generator(seed)
        return torch.asarray(rng.randn(shape), device=device)

    local_device = cpu if opts.randn_source == "CPU" or device.type == 'mps' else device
    local_generator = torch.Generator(local_device).manual_seed(int(seed))
    return torch.randn(shape, device=local_device, generator=local_generator).to(device)


def randn_like(x):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""

    from modules.shared import opts

    if opts.randn_source == "NV":
        return torch.asarray(nv_rng.randn(x.shape), device=x.device, dtype=x.dtype)

    if opts.randn_source == "CPU" or x.device.type == 'mps':
        return torch.randn_like(x, device=cpu).to(x.device)

    return torch.randn_like(x)


def randn_without_seed(shape):
    """Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.

    Use either randn() or manual_seed() to initialize the generator."""

    from modules.shared import opts

    if opts.randn_source == "NV":
        return torch.asarray(nv_rng.randn(shape), device=device)

    if opts.randn_source == "CPU" or device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)

    return torch.randn(shape, device=device)


def manual_seed(seed):
    """Set up a global random number generator using the specified seed."""
    from modules.shared import opts

    if opts.randn_source == "NV":
        global nv_rng
        nv_rng = rng_philox.Generator(seed)
        return

    torch.manual_seed(seed)


def autocast(disable=False):
    from modules import shared

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")


def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    from modules import shared

    if shared.cmd_opts.disable_nan_check:
        return

    if not torch.all(torch.isnan(x)).item():
        return

    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."

        if not shared.cmd_opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."

    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."

        if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."

    message += " Use --disable-nan-check commandline argument to disable this check."

    raise NansException(message)


@lru_cache
def first_time_calculation():
    """
    just do any calculation with pytorch layers - the first time this is done it allocaltes about 700MB of memory and
    spends about 2.7 seconds doing that, at least wih NVidia.
    """

    x = torch.zeros((1, 1)).to(device, dtype)
    linear = torch.nn.Linear(1, 1).to(device, dtype)
    linear(x)

    x = torch.zeros((1, 1, 3, 3)).to(device, dtype)
    conv2d = torch.nn.Conv2d(1, 1, (3, 3)).to(device, dtype)
    conv2d(x)
