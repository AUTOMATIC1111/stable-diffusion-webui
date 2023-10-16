# credits: @tfernd https://github.com/tfernd/HyperTile
# based on: https://github.com/tfernd/HyperTile/tree/main/hyper_tile/utils.py + https://github.com/tfernd/HyperTile/tree/main/hyper_tile/hyper_tile.py

from __future__ import annotations
from typing import Callable
from functools import wraps
from contextlib import contextmanager, nullcontext
import random
import math
import torch
import torch.nn as nn
from einops import rearrange
from modules.shared import log


# global variables to keep track of changing image size in multiple passes
height = None
width = None
max_h = 0
max_w = 0
error_reported = False
reset_needed = False


def possible_tile_sizes(dimension: int, tile_size: int, min_tile_size: int, tile_options: int) -> list[int]:
    assert tile_options >= 1
    min_tile_size = min(min_tile_size, tile_size, dimension)
    # all divisors that are themselves divisible by 8 and give tile-size above min
    n = torch.arange(1, dimension + 1)
    n = n[dimension // n // 8 * 8 * n == dimension]
    n = n[dimension // n >= min_tile_size]
    pos = (dimension // n).sub(tile_size).abs().argsort()
    pos = pos[:tile_options]
    return n[pos].tolist()


def parse_list(x: list[int], /) -> str:
    if len(x) == 0:
        return str(x[0])
    return str(x)


@contextmanager
def split_attention(layer: nn.Module, tile_size: int=256, min_tile_size: int=256, swap_size: int=1, depth: int=0):
    # hijacks AttnBlock from ldm and attention from diffusers
    global reset_needed # pylint: disable=global-statement
    ar = height / width # Aspect ratio
    reset_needed = True
    nhs = possible_tile_sizes(height, tile_size, min_tile_size, swap_size) # possible sub-grids that fit into the image
    nws = possible_tile_sizes(width, tile_size, min_tile_size, swap_size)
    def reset_nhs():
        nonlocal nhs, ar
        ar = height / width # Aspect ratio
        nhs = possible_tile_sizes(height, tile_size, min_tile_size, swap_size)

    def reset_nws():
        nonlocal nws, ar
        ar = height / width # Aspect ratio
        nws = possible_tile_sizes(width, tile_size, min_tile_size, swap_size)

    def self_attn_forward(forward: Callable) -> Callable:
        @wraps(forward)
        def wrapper(*args, **kwargs):
            global height, width, max_h, max_w, reset_needed, error_reported # pylint: disable=global-statement
            x = args[0]
            try:
                nh = nhs[random.randint(0, len(nhs) - 1)]
                nw = nws[random.randint(0, len(nws) - 1)]
            except Exception as e:
                if not error_reported:
                    error_reported = True
                    log.error(f'Hypertile error: width={width} height={height} {e}')
                out = forward(x, *args[1:], **kwargs)
                return out
            if x.ndim == 4: # VAE
                # TODO hyperlink vae breaks for diffusers when using non-standard sizes
                if nh * nw > 1:
                    x = rearrange(x, "b c (nh h) (nw w) -> (b nh nw) c h w", nh=nh, nw=nw)
                out = forward(x, *args[1:], **kwargs)
                if nh * nw > 1:
                    out = rearrange(out, "(b nh nw) c h w -> b c (nh h) (nw w)", nh=nh, nw=nw)
            else: # Unet
                hw = x.size(1)
                h, w = round(math.sqrt(ar * hw)), round(math.sqrt(hw / ar))
                # dynamic height/width based on fact that first two forward calls contain actual height/width
                # and reset if latest hw is larger since we're never downscaling in 2nd pass
                if reset_needed:
                    reset_nhs()
                    reset_nws()
                    max_h = height
                    max_w = width
                    reset_needed = False
                else:
                    if h > max_h:
                        height = 8 * h
                        max_h = max(max_h, h)
                        reset_nhs()
                    if w > max_w:
                        width = 8 * w
                        max_w = max(max_w, w)
                        reset_nws()
                down_ratio = max(height // 8 // h, 1)
                curr_depth = round(math.log(down_ratio, 2))
                # scale-up the tile-size the deeper we go
                nh = max(1, nh // down_ratio)
                nw = max(1, nw // down_ratio)
                do_split = curr_depth <= depth and h % nh == 0 and w % nw == 0 and nh * nw > 1
                try:
                    if do_split:
                        x = rearrange(x, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)
                    out = forward(x, *args[1:], **kwargs)
                    if do_split:
                        out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                        out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)
                except Exception as e:
                    if not error_reported:
                        error_reported = True
                        log.error(f'Hypertile error: width={width} height={height} {e}')
                    out = forward(x, *args[1:], **kwargs)
            return out
        return wrapper
    try: # hijack forward method and restore
        for name, module in layer.named_modules():
            if module.__class__.__qualname__ in ("Attention", "CrossAttention", "AttnBlock"):
                if name.endswith("attn2") or name.endswith("attn_2"): # skip cross-attention layers
                    continue
                setattr(module, "_original_forward", module.forward) # save original forward for recovery later # noqa: B010
                setattr(module, "forward", self_attn_forward(module.forward)) # noqa: B010
        yield
    finally:
        for _name, module in layer.named_modules():
            if hasattr(module, "_original_forward"): # remove hijack
                setattr(module, "forward", module._original_forward) # pylint: disable=protected-access # noqa: B010
                del module._original_forward


def context_hypertile_vae(p):
    global height, width, max_h, max_w, error_reported # pylint: disable=global-statement
    error_reported = False
    height=p.height
    width=p.width
    max_h = 0
    max_w = 0
    from modules import shared
    if p.sd_model is None or not shared.opts.hypertile_vae_enabled:
        return nullcontext()
    if shared.opts.cross_attention_optimization == 'Sub-quadratic':
        shared.log.warning('Hypertile UNet is not compatible with Sub-quadratic cross-attention optimization')
        return nullcontext()
    vae = getattr(p.sd_model, "vae", None) if shared.backend == shared.Backend.DIFFUSERS else getattr(p.sd_model, "first_stage_model", None)
    if height % 8 != 0 or width % 8 != 0:
        log.warning(f'Hypertile VAE disabled: width={width} height={height} are not divisible by 8')
        return nullcontext()
    if vae is None:
        shared.log.warning('Hypertile VAE is enabled but no VAE model was found')
        return nullcontext()
    else:
        shared.log.info(f'Applying hypertile: vae={shared.opts.hypertile_vae_tile}')
        p.extra_generation_params['Hypertile VAE'] = shared.opts.hypertile_vae_tile
        return split_attention(vae, tile_size=shared.opts.hypertile_vae_tile, min_tile_size=128, swap_size=1)


def context_hypertile_unet(p):
    global height, width, max_h, max_w, error_reported # pylint: disable=global-statement
    error_reported = False
    height=p.height
    width=p.width
    max_h = 0
    max_w = 0
    from modules import shared
    if p.sd_model is None or not shared.opts.hypertile_unet_enabled:
        return nullcontext()
    if shared.opts.cross_attention_optimization == 'Sub-quadratic' and not shared.cmd_opts.experimental:
        shared.log.warning('Hypertile UNet is not compatible with Sub-quadratic cross-attention optimization')
        return nullcontext()
    unet = getattr(p.sd_model, "unet", None) if shared.backend == shared.Backend.DIFFUSERS else getattr(p.sd_model.model, "diffusion_model", None)
    if height % 8 != 0 or width % 8 != 0:
        log.warning(f'Hypertile UNet disabled: width={width} height={height} are not divisible by 8')
        return nullcontext()
    if unet is None:
        shared.log.warning('Hypertile UNet is enabled but no Unet model was found')
        return nullcontext()
    else:
        shared.log.info(f'Applying hypertile: unet={shared.opts.hypertile_unet_tile}')
        p.extra_generation_params['Hypertile UNet'] = shared.opts.hypertile_unet_tile
        return split_attention(unet, tile_size=shared.opts.hypertile_unet_tile, min_tile_size=128, swap_size=1)


def hypertile_set(p, hr=False):
    from modules import shared
    global height, width, error_reported, reset_needed # pylint: disable=global-statement
    if not shared.opts.hypertile_unet_enabled:
        return
    error_reported = False
    height=p.height if not hr else getattr(p, 'hr_upscale_to_y', p.height)
    width=p.width if not hr else getattr(p, 'hr_upscale_to_x', p.width)
    reset_needed = True
