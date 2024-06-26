"""
Hypertile module for splitting attention layers in SD-1.5 U-Net and SD-1.5 VAE
Warn: The patch works well only if the input image has a width and height that are multiples of 128
Original author: @tfernd Github: https://github.com/tfernd/HyperTile
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from functools import wraps, cache

import math
import torch.nn as nn
import random

from einops import rearrange


@dataclass
class HypertileParams:
    depth = 0
    layer_name = ""
    tile_size: int = 0
    swap_size: int = 0
    aspect_ratio: float = 1.0
    forward = None
    enabled = False



# TODO add SD-XL layers
DEPTH_LAYERS = {
    0: [
        # SD 1.5 U-Net (diffusers)
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "input_blocks.1.1.transformer_blocks.0.attn1",
        "input_blocks.2.1.transformer_blocks.0.attn1",
        "output_blocks.9.1.transformer_blocks.0.attn1",
        "output_blocks.10.1.transformer_blocks.0.attn1",
        "output_blocks.11.1.transformer_blocks.0.attn1",
        # SD 1.5 VAE
        "decoder.mid_block.attentions.0",
        "decoder.mid.attn_1",
    ],
    1: [
        # SD 1.5 U-Net (diffusers)
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "input_blocks.4.1.transformer_blocks.0.attn1",
        "input_blocks.5.1.transformer_blocks.0.attn1",
        "output_blocks.6.1.transformer_blocks.0.attn1",
        "output_blocks.7.1.transformer_blocks.0.attn1",
        "output_blocks.8.1.transformer_blocks.0.attn1",
    ],
    2: [
        # SD 1.5 U-Net (diffusers)
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "input_blocks.7.1.transformer_blocks.0.attn1",
        "input_blocks.8.1.transformer_blocks.0.attn1",
        "output_blocks.3.1.transformer_blocks.0.attn1",
        "output_blocks.4.1.transformer_blocks.0.attn1",
        "output_blocks.5.1.transformer_blocks.0.attn1",
    ],
    3: [
        # SD 1.5 U-Net (diffusers)
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "middle_block.1.transformer_blocks.0.attn1",
    ],
}
# XL layers, thanks for GitHub@gel-crabs for the help
DEPTH_LAYERS_XL = {
    0: [
        # SD 1.5 U-Net (diffusers)
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "input_blocks.4.1.transformer_blocks.0.attn1",
        "input_blocks.5.1.transformer_blocks.0.attn1",
        "output_blocks.3.1.transformer_blocks.0.attn1",
        "output_blocks.4.1.transformer_blocks.0.attn1",
        "output_blocks.5.1.transformer_blocks.0.attn1",
        # SD 1.5 VAE
        "decoder.mid_block.attentions.0",
        "decoder.mid.attn_1",
    ],
    1: [
        # SD 1.5 U-Net (diffusers)
        #"down_blocks.1.attentions.0.transformer_blocks.0.attn1",
        #"down_blocks.1.attentions.1.transformer_blocks.0.attn1",
        #"up_blocks.2.attentions.0.transformer_blocks.0.attn1",
        #"up_blocks.2.attentions.1.transformer_blocks.0.attn1",
        #"up_blocks.2.attentions.2.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "input_blocks.4.1.transformer_blocks.1.attn1",
        "input_blocks.5.1.transformer_blocks.1.attn1",
        "output_blocks.3.1.transformer_blocks.1.attn1",
        "output_blocks.4.1.transformer_blocks.1.attn1",
        "output_blocks.5.1.transformer_blocks.1.attn1",
        "input_blocks.7.1.transformer_blocks.0.attn1",
        "input_blocks.8.1.transformer_blocks.0.attn1",
        "output_blocks.0.1.transformer_blocks.0.attn1",
        "output_blocks.1.1.transformer_blocks.0.attn1",
        "output_blocks.2.1.transformer_blocks.0.attn1",
        "input_blocks.7.1.transformer_blocks.1.attn1",
        "input_blocks.8.1.transformer_blocks.1.attn1",
        "output_blocks.0.1.transformer_blocks.1.attn1",
        "output_blocks.1.1.transformer_blocks.1.attn1",
        "output_blocks.2.1.transformer_blocks.1.attn1",
        "input_blocks.7.1.transformer_blocks.2.attn1",
        "input_blocks.8.1.transformer_blocks.2.attn1",
        "output_blocks.0.1.transformer_blocks.2.attn1",
        "output_blocks.1.1.transformer_blocks.2.attn1",
        "output_blocks.2.1.transformer_blocks.2.attn1",
        "input_blocks.7.1.transformer_blocks.3.attn1",
        "input_blocks.8.1.transformer_blocks.3.attn1",
        "output_blocks.0.1.transformer_blocks.3.attn1",
        "output_blocks.1.1.transformer_blocks.3.attn1",
        "output_blocks.2.1.transformer_blocks.3.attn1",
        "input_blocks.7.1.transformer_blocks.4.attn1",
        "input_blocks.8.1.transformer_blocks.4.attn1",
        "output_blocks.0.1.transformer_blocks.4.attn1",
        "output_blocks.1.1.transformer_blocks.4.attn1",
        "output_blocks.2.1.transformer_blocks.4.attn1",
        "input_blocks.7.1.transformer_blocks.5.attn1",
        "input_blocks.8.1.transformer_blocks.5.attn1",
        "output_blocks.0.1.transformer_blocks.5.attn1",
        "output_blocks.1.1.transformer_blocks.5.attn1",
        "output_blocks.2.1.transformer_blocks.5.attn1",
        "input_blocks.7.1.transformer_blocks.6.attn1",
        "input_blocks.8.1.transformer_blocks.6.attn1",
        "output_blocks.0.1.transformer_blocks.6.attn1",
        "output_blocks.1.1.transformer_blocks.6.attn1",
        "output_blocks.2.1.transformer_blocks.6.attn1",
        "input_blocks.7.1.transformer_blocks.7.attn1",
        "input_blocks.8.1.transformer_blocks.7.attn1",
        "output_blocks.0.1.transformer_blocks.7.attn1",
        "output_blocks.1.1.transformer_blocks.7.attn1",
        "output_blocks.2.1.transformer_blocks.7.attn1",
        "input_blocks.7.1.transformer_blocks.8.attn1",
        "input_blocks.8.1.transformer_blocks.8.attn1",
        "output_blocks.0.1.transformer_blocks.8.attn1",
        "output_blocks.1.1.transformer_blocks.8.attn1",
        "output_blocks.2.1.transformer_blocks.8.attn1",
        "input_blocks.7.1.transformer_blocks.9.attn1",
        "input_blocks.8.1.transformer_blocks.9.attn1",
        "output_blocks.0.1.transformer_blocks.9.attn1",
        "output_blocks.1.1.transformer_blocks.9.attn1",
        "output_blocks.2.1.transformer_blocks.9.attn1",
    ],
    2: [
        # SD 1.5 U-Net (diffusers)
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        # SD 1.5 U-Net (ldm)
        "middle_block.1.transformer_blocks.0.attn1",
        "middle_block.1.transformer_blocks.1.attn1",
        "middle_block.1.transformer_blocks.2.attn1",
        "middle_block.1.transformer_blocks.3.attn1",
        "middle_block.1.transformer_blocks.4.attn1",
        "middle_block.1.transformer_blocks.5.attn1",
        "middle_block.1.transformer_blocks.6.attn1",
        "middle_block.1.transformer_blocks.7.attn1",
        "middle_block.1.transformer_blocks.8.attn1",
        "middle_block.1.transformer_blocks.9.attn1",
    ],
    3 : [] # TODO - separate layers for SD-XL
}


RNG_INSTANCE = random.Random()

@cache
def get_divisors(value: int, min_value: int, /, max_options: int = 1) -> list[int]:
    """
    Returns divisors of value that
        x * min_value <= value
    in big -> small order, amount of divisors is limited by max_options
    """
    max_options = max(1, max_options) # at least 1 option should be returned
    min_value = min(min_value, value)
    divisors = [i for i in range(min_value, value + 1) if value % i == 0] # divisors in small -> big order
    ns = [value // i for i in divisors[:max_options]]  # has at least 1 element # big -> small order
    return ns


def random_divisor(value: int, min_value: int, /, max_options: int = 1) -> int:
    """
    Returns a random divisor of value that
        x * min_value <= value
    if max_options is 1, the behavior is deterministic
    """
    ns = get_divisors(value, min_value, max_options=max_options) # get cached divisors
    idx = RNG_INSTANCE.randint(0, len(ns) - 1)

    return ns[idx]


def set_hypertile_seed(seed: int) -> None:
    RNG_INSTANCE.seed(seed)


@cache
def largest_tile_size_available(width: int, height: int) -> int:
    """
    Calculates the largest tile size available for a given width and height
    Tile size is always a power of 2
    """
    gcd = math.gcd(width, height)
    largest_tile_size_available = 1
    while gcd % (largest_tile_size_available * 2) == 0:
        largest_tile_size_available *= 2
    return largest_tile_size_available


def iterative_closest_divisors(hw:int, aspect_ratio:float) -> tuple[int, int]:
    """
    Finds h and w such that h*w = hw and h/w = aspect_ratio
    We check all possible divisors of hw and return the closest to the aspect ratio
    """
    divisors = [i for i in range(2, hw + 1) if hw % i == 0] # all divisors of hw
    pairs = [(i, hw // i) for i in divisors] # all pairs of divisors of hw
    ratios = [w/h for h, w in pairs] # all ratios of pairs of divisors of hw
    closest_ratio = min(ratios, key=lambda x: abs(x - aspect_ratio)) # closest ratio to aspect_ratio
    closest_pair = pairs[ratios.index(closest_ratio)] # closest pair of divisors to aspect_ratio
    return closest_pair


@cache
def find_hw_candidates(hw:int, aspect_ratio:float) -> tuple[int, int]:
    """
    Finds h and w such that h*w = hw and h/w = aspect_ratio
    """
    h, w = round(math.sqrt(hw * aspect_ratio)), round(math.sqrt(hw / aspect_ratio))
    # find h and w such that h*w = hw and h/w = aspect_ratio
    if h * w != hw:
        w_candidate = hw / h
        # check if w is an integer
        if not w_candidate.is_integer():
            h_candidate = hw / w
            # check if h is an integer
            if not h_candidate.is_integer():
                return iterative_closest_divisors(hw, aspect_ratio)
            else:
                h = int(h_candidate)
        else:
            w = int(w_candidate)
    return h, w


def self_attn_forward(params: HypertileParams, scale_depth=True) -> Callable:

    @wraps(params.forward)
    def wrapper(*args, **kwargs):
        if not params.enabled:
            return params.forward(*args, **kwargs)

        latent_tile_size = max(128, params.tile_size) // 8
        x = args[0]

        # VAE
        if x.ndim == 4:
            b, c, h, w = x.shape

            nh = random_divisor(h, latent_tile_size, params.swap_size)
            nw = random_divisor(w, latent_tile_size, params.swap_size)

            if nh * nw > 1:
                x = rearrange(x, "b c (nh h) (nw w) -> (b nh nw) c h w", nh=nh, nw=nw)  # split into nh * nw tiles

            out = params.forward(x, *args[1:], **kwargs)

            if nh * nw > 1:
                out = rearrange(out, "(b nh nw) c h w -> b c (nh h) (nw w)", nh=nh, nw=nw)

        # U-Net
        else:
            hw: int = x.size(1)
            h, w = find_hw_candidates(hw, params.aspect_ratio)
            assert h * w == hw, f"Invalid aspect ratio {params.aspect_ratio} for input of shape {x.shape}, hw={hw}, h={h}, w={w}"

            factor = 2 ** params.depth if scale_depth else 1
            nh = random_divisor(h, latent_tile_size * factor, params.swap_size)
            nw = random_divisor(w, latent_tile_size * factor, params.swap_size)

            if nh * nw > 1:
                x = rearrange(x, "b (nh h nw w) c -> (b nh nw) (h w) c", h=h // nh, w=w // nw, nh=nh, nw=nw)

            out = params.forward(x, *args[1:], **kwargs)

            if nh * nw > 1:
                out = rearrange(out, "(b nh nw) hw c -> b nh nw hw c", nh=nh, nw=nw)
                out = rearrange(out, "b nh nw (h w) c -> b (nh h nw w) c", h=h // nh, w=w // nw)

        return out

    return wrapper


def hypertile_hook_model(model: nn.Module, width, height, *, enable=False, tile_size_max=128, swap_size=1, max_depth=3, is_sdxl=False):
    hypertile_layers = getattr(model, "__webui_hypertile_layers", None)
    if hypertile_layers is None:
        if not enable:
            return

        hypertile_layers = {}
        layers = DEPTH_LAYERS_XL if is_sdxl else DEPTH_LAYERS

        for depth in range(4):
            for layer_name, module in model.named_modules():
                if any(layer_name.endswith(try_name) for try_name in layers[depth]):
                    params = HypertileParams()
                    module.__webui_hypertile_params = params
                    params.forward = module.forward
                    params.depth = depth
                    params.layer_name = layer_name
                    module.forward = self_attn_forward(params)

                    hypertile_layers[layer_name] = 1

        model.__webui_hypertile_layers = hypertile_layers

    aspect_ratio = width / height
    tile_size = min(largest_tile_size_available(width, height), tile_size_max)

    for layer_name, module in model.named_modules():
        if layer_name in hypertile_layers:
            params = module.__webui_hypertile_params

            params.tile_size = tile_size
            params.swap_size = swap_size
            params.aspect_ratio = aspect_ratio
            params.enabled = enable and params.depth <= max_depth
