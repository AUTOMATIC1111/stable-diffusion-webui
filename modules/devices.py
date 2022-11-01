import sys, os, shlex
import contextlib
import torch
from modules import errors

# has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
has_mps = getattr(torch, 'has_mps', False)

cpu = torch.device("cpu")

def extract_device_id(args, name):
    for x in range(len(args)):
        if name in args[x]: return args[x+1]
    return None

def get_optimal_device():
    if torch.cuda.is_available():
        from modules import shared

        device_id = shared.cmd_opts.device_id

        if device_id is not None:
            cuda_device = f"cuda:{device_id}"
            return torch.device(cuda_device)
        else:
            return torch.device("cuda")

    if has_mps:
        return torch.device("mps")

    return cpu


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def enable_tf32():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


errors.run(enable_tf32, "Enabling TF32")

device = device_interrogate = device_gfpgan = device_swinir = device_esrgan = device_scunet = device_codeformer = None
dtype = torch.float16
dtype_vae = torch.float16

def randn(seed, shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    return torch.randn(shape, device=device)


def autocast(disable=False):
    from modules import shared

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")

# MPS workaround for https://github.com/pytorch/pytorch/issues/79383
def mps_contiguous(input_tensor, device): return input_tensor.contiguous() if device.type == 'mps' else input_tensor
def mps_contiguous_to(input_tensor, device): return mps_contiguous(input_tensor, device).to(device)
