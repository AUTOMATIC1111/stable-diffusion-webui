import torch
from packaging import version

from modules import devices, shared

import ldm.models.diffusion.ddpm
import ldm.modules.diffusionmodules.openaimodel
import ldm.modules.diffusionmodules.util
import ldm.modules.attention


class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    def __getattr__(self, item):
        if item == 'cat':
            return self.cat

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)


th = TorchHijackForUnet()


orig_apply_model = ldm.models.diffusion.ddpm.LatentDiffusion.apply_model
def apply_model(self, x_noisy, t, cond, **kwargs):
    if shared.cmd_opts.precision == "upcast" and devices.dtype == torch.float32 and devices.dtype_unet == torch.float16:
        cond['c_crossattn'] = [y.to(devices.dtype_unet) for y in cond['c_crossattn']]
        cond['c_concat'] = [y.to(devices.dtype_unet) for y in cond['c_concat']]
        return orig_apply_model(self, x_noisy.to(devices.dtype_unet), t.to(devices.dtype_unet), cond, **kwargs).to(devices.dtype)
    else:
        return orig_apply_model(self, x_noisy, t, cond, **kwargs)


ldm.models.diffusion.ddpm.LatentDiffusion.apply_model = apply_model


orig_timestep_embedding = ldm.modules.diffusionmodules.openaimodel.timestep_embedding
def timestep_embedding(*args, **kwargs):
    if shared.cmd_opts.precision == "upcast" and devices.dtype == torch.float32 and devices.dtype_unet == torch.float16:
        return orig_timestep_embedding(*args, **kwargs).to(devices.dtype_unet)
    else:
        return orig_timestep_embedding(*args, **kwargs)


ldm.modules.diffusionmodules.openaimodel.timestep_embedding = timestep_embedding


orig_GroupNorm32_forward = ldm.modules.diffusionmodules.util.GroupNorm32.forward
def GroupNorm32_forward(self, *args, **kwargs):
    if shared.cmd_opts.precision == "upcast" and devices.dtype == torch.float32 and devices.dtype_unet == torch.float16:
        return orig_GroupNorm32_forward(self.to(devices.dtype), *args, **kwargs)
    else:
        return orig_GroupNorm32_forward(self, *args, **kwargs)


orig_GEGLU_forward = ldm.modules.attention.GEGLU.forward
def GEGLU_forward(self, x):
    if shared.cmd_opts.precision == "upcast" and devices.dtype == torch.float32 and devices.dtype_unet == torch.float16:
        return orig_GEGLU_forward(self.to(devices.dtype), x.to(devices.dtype)).to(devices.dtype_unet)
    else:
        return orig_GEGLU_forward(self, x)


if version.parse(torch.__version__) <= version.parse("1.13.1"):
    ldm.modules.diffusionmodules.util.GroupNorm32.forward = GroupNorm32_forward
    ldm.modules.attention.GEGLU.forward = GEGLU_forward
