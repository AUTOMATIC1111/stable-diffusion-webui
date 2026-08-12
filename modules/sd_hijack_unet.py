import torch
from packaging import version
from einops import repeat
import math

from modules import devices, mps_fused_ops
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

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)


th = TorchHijackForUnet()


def fused_resblock_condition(_, self, x, emb):
    return (
        x.device.type == "mps"
        and x.dtype == torch.float16
        and x.ndim == 4
        and emb is not None
        and not self.training
        and not self.use_scale_shift_norm
        and not getattr(self, "skip_t_emb", False)
        and not getattr(self, "exchange_temb_dims", False)
        and len(self.in_layers) == 3
        and len(self.out_layers) == 4
        and isinstance(self.in_layers[0], torch.nn.GroupNorm)
        and isinstance(self.in_layers[1], torch.nn.SiLU)
        and isinstance(self.out_layers[0], torch.nn.GroupNorm)
        and isinstance(self.out_layers[1], torch.nn.SiLU)
    )


def fused_resblock_forward(_, self, x, emb):
    if self.updown:
        h = mps_fused_ops.group_norm_silu(x, self.in_layers[0])
        h = self.h_upd(h)
        x = self.x_upd(x)
        h = self.in_layers[2](h)
    else:
        h = self.in_layers[2](mps_fused_ops.group_norm_silu(x, self.in_layers[0]))

    emb_out = self.emb_layers(emb).type(h.dtype)
    h = mps_fused_ops.group_norm_silu_add_embedding(h, emb_out, self.out_layers[0])
    h = self.out_layers[2](h)
    h = self.out_layers[3](h)
    return self.skip_connection(x) + h


def fused_vae_resnet_condition(_, self, x, temb):
    return (
        x.device.type == "mps"
        and x.dtype == torch.float16
        and x.ndim == 4
        and not self.training
        and isinstance(self.norm1, torch.nn.GroupNorm)
        and isinstance(self.norm2, torch.nn.GroupNorm)
    )


def fused_vae_resnet_forward(_, self, x, temb):
    h = self.conv1(mps_fused_ops.group_norm_silu(x, self.norm1))

    if temb is not None:
        h = h + self.temb_proj(torch.nn.functional.silu(temb))[:, :, None, None]

    h = self.dropout(mps_fused_ops.group_norm_silu(h, self.norm2))
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)

    return x + h


def fused_geglu_condition(_, self, x):
    return (
        x.device.type == "mps"
        and x.dtype == torch.float16
        and x.ndim == 3
        and not self.training
        and hasattr(self, "proj")
    )


def fused_geglu_forward(_, self, x):
    return mps_fused_ops.geglu(x, self.proj)


# Below are monkey patches to enable upcasting a float16 UNet for float32 sampling
def apply_model(orig_func, self, x_noisy, t, cond, **kwargs):
    """Always make sure inputs to unet are in correct dtype."""
    if isinstance(cond, dict):
        for y in cond.keys():
            if isinstance(cond[y], list):
                cond[y] = [x.to(devices.dtype_unet) if isinstance(x, torch.Tensor) else x for x in cond[y]]
            else:
                cond[y] = cond[y].to(devices.dtype_unet) if isinstance(cond[y], torch.Tensor) else cond[y]

    with devices.autocast():
        result = orig_func(self, x_noisy.to(devices.dtype_unet), t.to(devices.dtype_unet), cond, **kwargs)
        if devices.unet_needs_upcast:
            return result.float()
        else:
            return result


# Monkey patch to create timestep embed tensor on device, avoiding a block.
def timestep_embedding(_, timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


# Monkey patch to SpatialTransformer removing unnecessary contiguous calls.
# Prevents a lot of unnecessary aten::copy_ calls
def spatial_transformer_forward(_, self, x: torch.Tensor, context=None):
    # note: if no context is given, cross-attention defaults to self-attention
    if not isinstance(context, list):
        context = [context]
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    if not self.use_linear:
        x = self.proj_in(x)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
    if self.use_linear:
        x = self.proj_in(x)
    for i, block in enumerate(self.transformer_blocks):
        x = block(x, context=context[i])
    if self.use_linear:
        x = self.proj_out(x)
    x = x.view(b, h, w, c).permute(0, 3, 1, 2)
    if not self.use_linear:
        x = self.proj_out(x)
    return x + x_in


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
        ddpm_edit_hijack = CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.apply_model', apply_model)


unet_needs_upcast = lambda *args, **kwargs: devices.unet_needs_upcast
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)
CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding', timestep_embedding)
CondFunc('ldm.modules.attention.SpatialTransformer.forward', spatial_transformer_forward)
CondFunc('ldm.modules.diffusionmodules.openaimodel.ResBlock._forward', fused_resblock_forward, fused_resblock_condition)
CondFunc('sgm.modules.diffusionmodules.openaimodel.ResBlock._forward', fused_resblock_forward, fused_resblock_condition)
CondFunc('ldm.modules.diffusionmodules.model.ResnetBlock.forward', fused_vae_resnet_forward, fused_vae_resnet_condition)
CondFunc('sgm.modules.diffusionmodules.model.ResnetBlock.forward', fused_vae_resnet_forward, fused_vae_resnet_condition)
CondFunc('ldm.modules.attention.GEGLU.forward', fused_geglu_forward, fused_geglu_condition)
CondFunc('sgm.modules.attention.GEGLU.forward', fused_geglu_forward, fused_geglu_condition)
CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding', lambda orig_func, timesteps, *args, **kwargs: orig_func(timesteps, *args, **kwargs).to(torch.float32 if timesteps.dtype == torch.int64 else devices.dtype_unet), unet_needs_upcast)

if version.parse(torch.__version__) <= version.parse("1.13.2") or torch.cuda.is_available():
    CondFunc('ldm.modules.diffusionmodules.util.GroupNorm32.forward', lambda orig_func, self, *args, **kwargs: orig_func(self.float(), *args, **kwargs), unet_needs_upcast)
    CondFunc('ldm.modules.attention.GEGLU.forward', lambda orig_func, self, x: orig_func(self.float(), x.float()).to(devices.dtype_unet), unet_needs_upcast)
    CondFunc('open_clip.transformer.ResidualAttentionBlock.__init__', lambda orig_func, *args, **kwargs: kwargs.update({'act_layer': GELUHijack}) and False or orig_func(*args, **kwargs), lambda _, *args, **kwargs: kwargs.get('act_layer') is None or kwargs['act_layer'] == torch.nn.GELU)

first_stage_cond = lambda _, self, *args, **kwargs: devices.unet_needs_upcast and self.model.diffusion_model.dtype == torch.float16
first_stage_sub = lambda orig_func, self, x, **kwargs: orig_func(self, x.to(devices.dtype_vae), **kwargs)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).float(), first_stage_cond)

CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.apply_model', apply_model)
CondFunc('sgm.modules.diffusionmodules.wrappers.OpenAIWrapper.forward', apply_model)


def timestep_embedding_cast_result(orig_func, timesteps, *args, **kwargs):
    if devices.unet_needs_upcast and timesteps.dtype == torch.int64:
        dtype = torch.float32
    else:
        dtype = devices.dtype_unet
    return orig_func(timesteps, *args, **kwargs).to(dtype=dtype)


CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding', timestep_embedding_cast_result)
CondFunc('sgm.modules.diffusionmodules.openaimodel.timestep_embedding', timestep_embedding_cast_result)
