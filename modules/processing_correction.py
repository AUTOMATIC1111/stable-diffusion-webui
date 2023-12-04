"""
based on article by TimothyAlexisVass
https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
"""

import os
import torch
from modules import shared


debug = shared.log.info if os.environ.get('SD_HDR_DEBUG', None) is not None else lambda *args, **kwargs: None


def soft_clamp_tensor(input_tensor, threshold=0.8, boundary=4):
    # shrinking towards the mean; will also remove outliers
    if max(abs(input_tensor.max()), abs(input_tensor.min())) < boundary or threshold == 0:
        return input_tensor
    channel_dim = 1
    threshold *= boundary
    max_vals = input_tensor.max(channel_dim, keepdim=True)[0]
    max_replace = ((input_tensor - threshold) / (max_vals - threshold)) * (boundary - threshold) + threshold
    over_mask = input_tensor > threshold
    min_vals = input_tensor.min(channel_dim, keepdim=True)[0]
    min_replace = ((input_tensor + threshold) / (min_vals + threshold)) * (-boundary + threshold) - threshold
    under_mask = input_tensor < -threshold
    debug(f'HDE soft clamp: threshold={threshold} boundary={boundary}')
    input_tensor = torch.where(over_mask, max_replace, torch.where(under_mask, min_replace, input_tensor))
    return input_tensor


def center_tensor(input_tensor, channel_shift=1.0, full_shift=1.0, channels=[0, 1, 2, 3]): # pylint: disable=dangerous-default-value # noqa: B006
    if channel_shift == 0 and full_shift == 0:
        return input_tensor
    means = []
    for channel in channels:
        means.append(input_tensor[0, channel].mean())
        input_tensor[0, channel] -= means[-1] * channel_shift
    debug(f'HDR center: channel-shift{channel_shift} full-shift={full_shift} means={torch.stack(means)}')
    input_tensor = input_tensor - input_tensor.mean() * full_shift
    return input_tensor


def maximize_tensor(input_tensor, boundary=1.0, channels=[0, 1, 2]): # pylint: disable=dangerous-default-value # noqa: B006
    if boundary == 1.0:
        return input_tensor
    boundary *= 4
    min_val = input_tensor.min()
    max_val = input_tensor.max()
    normalization_factor = boundary / max(abs(min_val), abs(max_val))
    input_tensor[0, channels] *= normalization_factor
    debug(f'HDR maximize: boundary={boundary} min={min_val} max={max_val} factor={normalization_factor}')
    return input_tensor


def correction_callback(p, timestep, kwags):
    if timestep > 950 and p.hdr_clamp:
        kwags["latents"] = soft_clamp_tensor(kwags["latents"], threshold=p.hdr_threshold, boundary=p.hdr_boundary)
    if timestep > 700 and p.hdr_center:
        kwags["latents"] = center_tensor(kwags["latents"], channel_shift=p.hdr_channel_shift, full_shift=p.hdr_full_shift)
    if timestep > 1 and timestep < 100 and p.hdr_maximize:
        kwags["latents"] = center_tensor(kwags["latents"], channel_shift=p.hdr_max_center, full_shift=1.0)
        kwags["latents"] = maximize_tensor(kwags["latents"], boundary=p.hdr_max_boundry)
    return kwags
