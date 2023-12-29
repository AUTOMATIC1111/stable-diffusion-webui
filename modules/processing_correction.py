"""
based on article by TimothyAlexisVass
https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
"""

import os
import torch
from modules import shared


debug = shared.log.trace if os.environ.get('SD_HDR_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: HDR')


def soft_clamp_tensor(tensor, threshold=0.8, boundary=4):
    # shrinking towards the mean; will also remove outliers
    if max(abs(tensor.max()), abs(tensor.min())) < boundary or threshold == 0:
        return tensor
    channel_dim = 0
    threshold *= boundary
    max_vals = tensor.max(channel_dim, keepdim=True)[0]
    max_replace = ((tensor - threshold) / (max_vals - threshold)) * (boundary - threshold) + threshold
    over_mask = tensor > threshold
    min_vals = tensor.min(channel_dim, keepdim=True)[0]
    min_replace = ((tensor + threshold) / (min_vals + threshold)) * (-boundary + threshold) - threshold
    under_mask = tensor < -threshold
    tensor = torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, tensor))
    debug(f'HDR soft clamp: threshold={threshold} boundary={boundary} shape={tensor.shape}')
    return tensor


def center_tensor(tensor, channel_shift=1.0, full_shift=1.0, channels=[0, 1, 2, 3]): # pylint: disable=dangerous-default-value # noqa: B006
    if channel_shift == 0 and full_shift == 0:
        return tensor
    means = []
    for channel in channels:
        means.append(tensor[0, channel].mean())
        # tensor[0, channel] -= means[-1] * channel_shift
        tensor[channel] -= means[-1] * channel_shift
    tensor = tensor - tensor.mean() * full_shift
    debug(f'HDR center: channel-shift={channel_shift} full-shift={full_shift} means={torch.stack(means)} shape={tensor.shape}')
    return tensor


def maximize_tensor(tensor, boundary=1.0, _channels=[0, 1, 2]): # pylint: disable=dangerous-default-value # noqa: B006
    if boundary == 1.0:
        return tensor
    boundary *= 4
    min_val = tensor.min()
    max_val = tensor.max()
    normalization_factor = boundary / max(abs(min_val), abs(max_val))
    # tensor[0, channels] *= normalization_factor
    tensor *= normalization_factor
    debug(f'HDR maximize: boundary={boundary} min={min_val} max={max_val} factor={normalization_factor} shape={tensor.shape}')
    return tensor


def correction(p, timestep, latent):
    if timestep > 950 and p.hdr_clamp:
        p.extra_generation_params["HDR clamp"] = f'{p.hdr_threshold}/{p.hdr_boundary}'
        latent = soft_clamp_tensor(latent, threshold=p.hdr_threshold, boundary=p.hdr_boundary)
    if timestep > 700 and p.hdr_center:
        p.extra_generation_params["HDR center"] = f'{p.hdr_channel_shift}/{p.hdr_full_shift}'
        latent = center_tensor(latent, channel_shift=p.hdr_channel_shift, full_shift=p.hdr_full_shift)
    if timestep > 1 and timestep < 100 and p.hdr_maximize:
        p.extra_generation_params["HDR max"] = f'{p.hdr_max_center}/p.hdr_max_boundry'
        latent = center_tensor(latent, channel_shift=p.hdr_max_center, full_shift=1.0)
        latent = maximize_tensor(latent, boundary=p.hdr_max_boundry)
    return latent


def correction_callback(p, timestep, kwargs):
    if not p.hdr_clamp and not p.hdr_center and not p.hdr_maximize:
        return kwargs
    latents = kwargs["latents"]
    # debug(f'HDR correction: latents={latents.shape}')
    if len(latents.shape) == 4: # standard batched latent
        for i in range(latents.shape[0]):
            latents[i] = correction(p, timestep, latents[i])
    elif len(latents.shape) == 5 and latents.shape[0] == 1: # probably animatediff
        latents = latents.squeeze(0).permute(1, 0, 2, 3)
        for i in range(latents.shape[0]):
            latents[i] = correction(p, timestep, latents[i])
        latents = latents.permute(1, 0, 2, 3).unsqueeze(0)
    else:
        shared.log.debug(f'HDR correction: unknown latent shape {latents.shape}')
    kwargs["latents"] = latents
    return kwargs
