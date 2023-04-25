"""
MIT License

Copyright (c) 2023

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import math
import torch
import gradio as gr
import torch.nn.functional as F
import modules.scripts as scripts

from torch import nn, einsum
from inspect import isfunction
from einops import rearrange, repeat

from modules import shared
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import on_cfg_denoiser, CFGDenoiserParams, CFGDenoisedParams, on_cfg_denoised, \
    AfterCFGCallbackParams, on_cfg_after_cfg

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class LoggedSelfAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.attn_probs = None

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with torch.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        self.attn_probs = sim

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


def xattn_forward_log(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION == "fp32":
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    del q, k

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    self.attn_probs = sim
    global current_selfattn_map
    current_selfattn_map = sim

    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = self.to_out(out)
    global current_outsize
    current_outsize = out.shape[-2:]
    return out


saved_original_selfattn_forward = None
current_selfattn_map = None
current_sag_guidance_scale = 1.0
sag_enabled = False
sag_mask_threshold = 1.0

current_xin = None
current_outsize = (64, 64)
current_batch_size = 1
current_degraded_pred = None
current_unet_kwargs = {}
current_uncond_pred = None
current_degraded_pred_compensation = None


def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class Script(scripts.Script):

    def __init__(self):
        pass

    def title(self):
        return "Self Attention Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def denoiser_callback(self, parms: CFGDenoiserParams):
        if not sag_enabled:
            return
        global current_xin, current_batch_size

        # logging current uncond size for cond/uncond output separation
        current_batch_size = parms.text_uncond.shape[0]
        # logging current input for eps calculation later
        current_xin = parms.x[-current_batch_size:]

        # logging necessary information for SAG pred
        current_uncond_emb = parms.text_uncond
        current_sigma = parms.sigma
        current_image_cond_in = parms.image_cond
        global current_unet_kwargs
        current_unet_kwargs = {
            "sigma": current_sigma[-current_batch_size:],
            "image_cond": current_image_cond_in[-current_batch_size:],
            "text_uncond": current_uncond_emb,
        }

    def denoised_callback(self, params: CFGDenoisedParams):
        if not sag_enabled:
            return
        # output from DiscreteEpsDDPMDenoiser is already pred_x0
        uncond_output = params.x[-current_batch_size:]
        original_latents = uncond_output
        global current_uncond_pred
        current_uncond_pred = uncond_output

        # Produce attention mask
        # We're only interested in the last current_batch_size*head_count slices of logged self-attention map
        attn_map = current_selfattn_map[-current_batch_size * 8:]
        bh, hw1, hw2 = attn_map.shape
        b, latent_channel, latent_h, latent_w = original_latents.shape
        h = 8

        middle_layer_latent_size = [math.ceil(latent_h / 8), math.ceil(latent_w / 8)]

        attn_map = attn_map.reshape(b, h, hw1, hw2)
        attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > sag_mask_threshold
        attn_mask = (
            attn_mask.reshape(b, middle_layer_latent_size[0], middle_layer_latent_size[1])
            .unsqueeze(1)
            .repeat(1, latent_channel, 1, 1)
            .type(attn_map.dtype)
        )
        attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

        # Blur according to the self-attention mask
        degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
        degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)

        renoised_degraded_latent = degraded_latents - (uncond_output - current_xin)
        # renoised_degraded_latent = degraded_latents
        # get predicted x0 in degraded direction
        global current_degraded_pred_compensation
        current_degraded_pred_compensation = uncond_output - degraded_latents
        if shared.sd_model.model.conditioning_key == "crossattn-adm":
            make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": c_crossattn, "c_adm": c_adm}
        else:
            make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": c_crossattn, "c_concat": [c_concat]}
        degraded_pred = params.inner_model(
            renoised_degraded_latent, current_unet_kwargs['sigma'],
            cond=make_condition_dict([current_unet_kwargs['text_uncond']], [current_unet_kwargs['image_cond']]))
        global current_degraded_pred
        current_degraded_pred = degraded_pred

    def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams):
        if not sag_enabled:
            return

        params.x = params.x + (
                current_uncond_pred - (current_degraded_pred + current_degraded_pred_compensation)) * float(
            current_sag_guidance_scale)
        params.output_altered = True

    def ui(self, is_img2img):
        with gr.Accordion('Self Attention Guidance', open=False):
            enabled = gr.Checkbox(label="Enabled", default=False)
            scale = gr.Slider(label='Scale', minimum=-2.0, maximum=10.0, step=0.05, value=0.75)
            mask_threshold = gr.Slider(label='SAG Mask Threshold', minimum=0.0, maximum=2.0, step=0.05, value=1.0)

        return [enabled, scale, mask_threshold]

    def process(self, p: StableDiffusionProcessing, *args, **kwargs):
        enabled, scale, mask_threshold = args
        global sag_enabled, sag_mask_threshold
        if enabled:

            sag_enabled = True
            sag_mask_threshold = mask_threshold
            global current_sag_guidance_scale
            current_sag_guidance_scale = scale
            global saved_original_selfattn_forward
            # replace target self attention module in unet with ours

            org_attn_module = \
                shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
            saved_original_selfattn_forward = org_attn_module.forward
            org_attn_module.forward = xattn_forward_log.__get__(org_attn_module, org_attn_module.__class__)

            p.extra_generation_params["SAG Guidance Scale"] = scale
            p.extra_generation_params["SAG Mask Threshold"] = mask_threshold

        else:
            sag_enabled = False

        if not hasattr(self, 'callbacks_added'):
            on_cfg_denoiser(self.denoiser_callback)
            on_cfg_denoised(self.denoised_callback)
            on_cfg_after_cfg(self.cfg_after_cfg_callback)
            self.callbacks_added = True

        return

    def postprocess(self, p, processed, *args):
        enabled, scale, sag_mask_threshold = args
        if enabled:
            # restore original self attention module forward function
            attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules[
                '0'].attn1
            attn_module.forward = saved_original_selfattn_forward
        return
