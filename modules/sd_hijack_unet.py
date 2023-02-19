import torch
from packaging import version

from modules import devices
from modules.sd_hijack_utils import CondFunc


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


# Below are monkey patches to enable upcasting a float16 UNet for float32 sampling
def apply_model(orig_func, self, x_noisy, t, cond, **kwargs):

    if isinstance(cond, dict):
        for y in cond.keys():
            cond[y] = [x.to(devices.dtype_unet) if isinstance(x, torch.Tensor) else x for x in cond[y]]

    with devices.autocast():
        return orig_func(self, x_noisy.to(devices.dtype_unet), t.to(devices.dtype_unet), cond, **kwargs).float()


class GELUHijack(torch.nn.GELU, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.GELU.__init__(self, *args, **kwargs)
    def forward(self, x):
        if devices.unet_needs_upcast:
            return torch.nn.GELU.forward(self.float(), x.float()).to(devices.dtype_unet)
        else:
            return torch.nn.GELU.forward(self, x)


ddpm_edit_hijack = None
def hijack_ddpm_edit():
    global ddpm_edit_hijack
    if not ddpm_edit_hijack:
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
        ddpm_edit_hijack = CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)


unet_needs_upcast = lambda *args, **kwargs: devices.unet_needs_upcast
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)
CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding', lambda orig_func, timesteps, *args, **kwargs: orig_func(timesteps, *args, **kwargs).to(torch.float32 if timesteps.dtype == torch.int64 else devices.dtype_unet), unet_needs_upcast)
if version.parse(torch.__version__) <= version.parse("1.13.1"):
    CondFunc('ldm.modules.diffusionmodules.util.GroupNorm32.forward', lambda orig_func, self, *args, **kwargs: orig_func(self.float(), *args, **kwargs), unet_needs_upcast)
    CondFunc('ldm.modules.attention.GEGLU.forward', lambda orig_func, self, x: orig_func(self.float(), x.float()).to(devices.dtype_unet), unet_needs_upcast)
    CondFunc('open_clip.transformer.ResidualAttentionBlock.__init__', lambda orig_func, *args, **kwargs: kwargs.update({'act_layer': GELUHijack}) and False or orig_func(*args, **kwargs), lambda _, *args, **kwargs: kwargs.get('act_layer') is None or kwargs['act_layer'] == torch.nn.GELU)

first_stage_cond = lambda _, self, *args, **kwargs: devices.unet_needs_upcast and self.model.diffusion_model.dtype == torch.float16
first_stage_sub = lambda orig_func, self, x, **kwargs: orig_func(self, x.to(devices.dtype_vae), **kwargs)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).float(), first_stage_cond)
