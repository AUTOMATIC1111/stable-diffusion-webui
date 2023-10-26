# Diffusers 0.10.2からStable Diffusionに必要な部分だけを持ってくる
# 条件分岐等で不要な部分は削除している
# コードの多くはDiffusersからコピーしている
# 制約として、モデルのstate_dictがDiffusers 0.10.2のものと同じ形式である必要がある

# Copy from Diffusers 0.10.2 for Stable Diffusion. Most of the code is copied from Diffusers.
# Unnecessary parts are deleted by condition branching.
# As a constraint, the state_dict of the model must be in the same format as that of Diffusers 0.10.2

"""
v1.5とv2.1の相違点は
- attention_head_dimがintかlist[int]か
- cross_attention_dimが768か1024か
- use_linear_projection: trueがない（=False, 1.5）かあるか
- upcast_attentionがFalse(1.5)かTrue(2.1)か
- （以下は多分無視していい）
- sample_sizeが64か96か
- dual_cross_attentionがあるかないか
- num_class_embedsがあるかないか
- only_cross_attentionがあるかないか

v1.5
{
  "_class_name": "UNet2DConditionModel",
  "_diffusers_version": "0.6.0",
  "act_fn": "silu",
  "attention_head_dim": 8,
  "block_out_channels": [
    320,
    640,
    1280,
    1280
  ],
  "center_input_sample": false,
  "cross_attention_dim": 768,
  "down_block_types": [
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "DownBlock2D"
  ],
  "downsample_padding": 1,
  "flip_sin_to_cos": true,
  "freq_shift": 0,
  "in_channels": 4,
  "layers_per_block": 2,
  "mid_block_scale_factor": 1,
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
  "out_channels": 4,
  "sample_size": 64,
  "up_block_types": [
    "UpBlock2D",
    "CrossAttnUpBlock2D",
    "CrossAttnUpBlock2D",
    "CrossAttnUpBlock2D"
  ]
}

v2.1
{
  "_class_name": "UNet2DConditionModel",
  "_diffusers_version": "0.10.0.dev0",
  "act_fn": "silu",
  "attention_head_dim": [
    5,
    10,
    20,
    20
  ],
  "block_out_channels": [
    320,
    640,
    1280,
    1280
  ],
  "center_input_sample": false,
  "cross_attention_dim": 1024,
  "down_block_types": [
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "CrossAttnDownBlock2D",
    "DownBlock2D"
  ],
  "downsample_padding": 1,
  "dual_cross_attention": false,
  "flip_sin_to_cos": true,
  "freq_shift": 0,
  "in_channels": 4,
  "layers_per_block": 2,
  "mid_block_scale_factor": 1,
  "norm_eps": 1e-05,
  "norm_num_groups": 32,
  "num_class_embeds": null,
  "only_cross_attention": false,
  "out_channels": 4,
  "sample_size": 96,
  "up_block_types": [
    "UpBlock2D",
    "CrossAttnUpBlock2D",
    "CrossAttnUpBlock2D",
    "CrossAttnUpBlock2D"
  ],
  "use_linear_projection": true,
  "upcast_attention": true
}
"""

import math
from types import SimpleNamespace
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

BLOCK_OUT_CHANNELS: Tuple[int] = (320, 640, 1280, 1280)
TIMESTEP_INPUT_DIM = BLOCK_OUT_CHANNELS[0]
TIME_EMBED_DIM = BLOCK_OUT_CHANNELS[0] * 4
IN_CHANNELS: int = 4
OUT_CHANNELS: int = 4
LAYERS_PER_BLOCK: int = 2
LAYERS_PER_BLOCK_UP: int = LAYERS_PER_BLOCK + 1
TIME_EMBED_FLIP_SIN_TO_COS: bool = True
TIME_EMBED_FREQ_SHIFT: int = 0
NORM_GROUPS: int = 32
NORM_EPS: float = 1e-5
TRANSFORMER_NORM_NUM_GROUPS = 32

DOWN_BLOCK_TYPES = ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"]
UP_BLOCK_TYPES = ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"]


# region memory efficient attention

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# flash attention forwards and backwards

# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """Algorithm 2 in the paper"""

        device = q.device
        dtype = q.dtype
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), dtype=dtype, device=device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

        scale = q.shape[-1] ** -0.5

        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, "b n -> b 1 1 n")
            mask = mask.split(q_bucket_size, dim=-1)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = torch.einsum("... i d, ... j d -> ... i j", qc, kc) * scale

                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(
                        q_start_index - k_start_index + 1
                    )
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)

                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.0)

                block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_values = torch.einsum("... i j, ... j d -> ... i d", exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """Algorithm 4 in the paper"""

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, l, m = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            do.split(q_bucket_size, dim=-2),
            mask,
            l.split(q_bucket_size, dim=-2),
            m.split(q_bucket_size, dim=-2),
            dq.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                dk.split(k_bucket_size, dim=-2),
                dv.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = torch.einsum("... i d, ... j d -> ... i j", qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(
                        q_start_index - k_start_index + 1
                    )
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                exp_attn_weights = torch.exp(attn_weights - mc)

                if exists(row_mask):
                    exp_attn_weights.masked_fill_(~row_mask, 0.0)

                p = exp_attn_weights / lc

                dv_chunk = torch.einsum("... i j, ... i d -> ... j d", p, doc)
                dp = torch.einsum("... i d, ... j d -> ... i j", doc, vc)

                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds = p * scale * (dp - D)

                dq_chunk = torch.einsum("... i j, ... j d -> ... i d", ds, kc)
                dk_chunk = torch.einsum("... i j, ... i d -> ... j d", ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None


# endregion


def get_parameter_dtype(parameter: torch.nn.Module):
    return next(parameter.parameters()).dtype


def get_parameter_device(parameter: torch.nn.Module):
    return next(parameter.parameters()).device


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class SampleOutput:
    def __init__(self, sample):
        self.sample = sample


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu", out_dim: int = None):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

    def forward(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = torch.nn.GroupNorm(num_groups=NORM_GROUPS, num_channels=in_channels, eps=NORM_EPS, affine=True)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = torch.nn.Linear(TIME_EMBED_DIM, out_channels)

        self.norm2 = torch.nn.GroupNorm(num_groups=NORM_GROUPS, num_channels=out_channels, eps=NORM_EPS, affine=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # if non_linearity == "swish":
        self.nonlinearity = lambda x: F.silu(x)

        self.use_in_shortcut = self.in_channels != self.out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_downsample=True,
    ):
        super().__init__()

        self.has_cross_attention = False
        resnets = []

        for i in range(LAYERS_PER_BLOCK):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = [Downsample2D(out_channels, out_channels=out_channels)]
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        pass

    def set_use_sdpa(self, sdpa):
        pass

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class Downsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        # no dropout here

        self.use_memory_efficient_attention_xformers = False
        self.use_memory_efficient_attention_mem_eff = False
        self.use_sdpa = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        self.use_memory_efficient_attention_xformers = xformers
        self.use_memory_efficient_attention_mem_eff = mem_eff

    def set_use_sdpa(self, sdpa):
        self.use_sdpa = sdpa

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        if self.use_memory_efficient_attention_xformers:
            return self.forward_memory_efficient_xformers(hidden_states, context, mask)
        if self.use_memory_efficient_attention_mem_eff:
            return self.forward_memory_efficient_mem_eff(hidden_states, context, mask)
        if self.use_sdpa:
            return self.forward_sdpa(hidden_states, context, mask)

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        hidden_states = self._attention(query, key, value)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # hidden_states = self.to_out[1](hidden_states)     # no dropout
        return hidden_states

    def _attention(self, query, key, value):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    # TODO support Hypernetworks
    def forward_memory_efficient_xformers(self, x, context=None, mask=None):
        import xformers.ops

        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

        out = rearrange(out, "b n h d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out

    def forward_memory_efficient_mem_eff(self, x, context=None, mask=None):
        flash_func = FlashAttentionFunction

        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k = self.to_k(context)
        v = self.to_v(context)
        del context, x

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out[0](out)
        return out

    def forward_sdpa(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)
        context = context if context is not None else x
        context = context.to(x.dtype)
        k_in = self.to_k(context)
        v_in = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        out = self.to_out[0](out)
        return out


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        inner_dim = int(dim * 4)  # mult is always 4

        self.net = nn.ModuleList([])
        # project in
        self.net.append(GEGLU(dim, inner_dim))
        # project dropout
        self.net.append(nn.Identity())  # nn.Dropout(0)) # dummy for dropout with 0
        # project out
        self.net.append(nn.Linear(inner_dim, dim))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, cross_attention_dim: int, upcast_attention: bool = False
    ):
        super().__init__()

        # 1. Self-Attn
        self.attn1 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )
        self.ff = FeedForward(dim)

        # 2. Cross-Attn
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool):
        self.attn1.set_use_memory_efficient_attention(xformers, mem_eff)
        self.attn2.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool):
        self.attn1.set_use_sdpa(sdpa)
        self.attn2.set_use_sdpa(sdpa)

    def forward(self, hidden_states, context=None, timestep=None):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


class Transformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.use_linear_projection = use_linear_projection

        self.norm = torch.nn.GroupNorm(num_groups=TRANSFORMER_NORM_NUM_GROUPS, num_channels=in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                )
            ]
        )

        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        for transformer in self.transformer_blocks:
            transformer.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa):
        for transformer in self.transformer_blocks:
            transformer.set_use_sdpa(sdpa)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # 1. Input
        batch, _, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=encoder_hidden_states, timestep=timestep)

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return SampleOutput(sample=output)


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_downsample=True,
        cross_attention_dim=1280,
        attn_num_head_channels=1,
        use_linear_projection=False,
        upcast_attention=False,
    ):
        super().__init__()
        self.has_cross_attention = True
        resnets = []
        attentions = []

        self.attn_num_head_channels = attn_num_head_channels

        for i in range(LAYERS_PER_BLOCK):
            in_channels = in_channels if i == 0 else out_channels

            resnets.append(ResnetBlock2D(in_channels=in_channels, out_channels=out_channels))
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        for attn in self.attentions:
            attn.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa):
        for attn in self.attentions:
            attn.set_use_sdpa(sdpa)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        use_linear_projection=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        # Middle block has two resnets and one attention
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
            ),
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
            ),
        ]
        attentions = [
            Transformer2DModel(
                attn_num_head_channels,
                in_channels // attn_num_head_channels,
                in_channels=in_channels,
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=use_linear_projection,
            )
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        for attn in self.attentions:
            attn.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa):
        for attn in self.attentions:
            attn.set_use_sdpa(sdpa)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        for i, resnet in enumerate(self.resnets):
            attn = None if i == 0 else self.attentions[i - 1]

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                if attn is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states
                    )[0]

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                if attn is not None:
                    hidden_states = attn(hidden_states, encoder_hidden_states).sample
                hidden_states = resnet(hidden_states, temb)

        return hidden_states


class Upsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        hidden_states = self.conv(hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        add_upsample=True,
    ):
        super().__init__()

        self.has_cross_attention = False
        resnets = []

        for i in range(LAYERS_PER_BLOCK_UP):
            res_skip_channels = in_channels if (i == LAYERS_PER_BLOCK_UP - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        pass

    def set_use_sdpa(self, sdpa):
        pass

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        add_upsample=True,
        use_linear_projection=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(LAYERS_PER_BLOCK_UP):
            res_skip_channels = in_channels if (i == LAYERS_PER_BLOCK_UP - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                )
            )
            attentions.append(
                Transformer2DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    cross_attention_dim=cross_attention_dim,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        for attn in self.attentions:
            attn.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, spda):
        for attn in self.attentions:
            attn.set_use_sdpa(spda)

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False), hidden_states, encoder_hidden_states
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


def get_down_block(
    down_block_type,
    in_channels,
    out_channels,
    add_downsample,
    attn_num_head_channels,
    cross_attention_dim,
    use_linear_projection,
    upcast_attention,
):
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        return CrossAttnDownBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )


def get_up_block(
    up_block_type,
    in_channels,
    out_channels,
    prev_output_channel,
    add_upsample,
    attn_num_head_channels,
    cross_attention_dim=None,
    use_linear_projection=False,
    upcast_attention=False,
):
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            add_upsample=add_upsample,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        return CrossAttnUpBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            attn_num_head_channels=attn_num_head_channels,
            cross_attention_dim=cross_attention_dim,
            add_upsample=add_upsample,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )


class UNet2DConditionModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        sample_size: Optional[int] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        cross_attention_dim: int = 1280,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert sample_size is not None, "sample_size must be specified"
        print(
            f"UNet2DConditionModel: {sample_size}, {attention_head_dim}, {cross_attention_dim}, {use_linear_projection}, {upcast_attention}"
        )

        # 外部からの参照用に定義しておく
        self.in_channels = IN_CHANNELS
        self.out_channels = OUT_CHANNELS

        self.sample_size = sample_size
        self.prepare_config()

        # state_dictの書式が変わるのでmoduleの持ち方は変えられない

        # input
        self.conv_in = nn.Conv2d(IN_CHANNELS, BLOCK_OUT_CHANNELS[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(BLOCK_OUT_CHANNELS[0], TIME_EMBED_FLIP_SIN_TO_COS, TIME_EMBED_FREQ_SHIFT)

        self.time_embedding = TimestepEmbedding(TIMESTEP_INPUT_DIM, TIME_EMBED_DIM)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * 4

        # down
        output_channel = BLOCK_OUT_CHANNELS[0]
        for i, down_block_type in enumerate(DOWN_BLOCK_TYPES):
            input_channel = output_channel
            output_channel = BLOCK_OUT_CHANNELS[i]
            is_final_block = i == len(BLOCK_OUT_CHANNELS) - 1

            down_block = get_down_block(
                down_block_type,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                attn_num_head_channels=attention_head_dim[i],
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=BLOCK_OUT_CHANNELS[-1],
            attn_num_head_channels=attention_head_dim[-1],
            cross_attention_dim=cross_attention_dim,
            use_linear_projection=use_linear_projection,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(BLOCK_OUT_CHANNELS))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(UP_BLOCK_TYPES):
            is_final_block = i == len(BLOCK_OUT_CHANNELS) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(BLOCK_OUT_CHANNELS) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                add_upsample=add_upsample,
                attn_num_head_channels=reversed_attention_head_dim[i],
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=BLOCK_OUT_CHANNELS[0], num_groups=NORM_GROUPS, eps=NORM_EPS)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(BLOCK_OUT_CHANNELS[0], OUT_CHANNELS, kernel_size=3, padding=1)

    # region diffusers compatibility
    def prepare_config(self):
        self.config = SimpleNamespace()

    @property
    def dtype(self) -> torch.dtype:
        # `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        return get_parameter_dtype(self)

    @property
    def device(self) -> torch.device:
        # `torch.device`: The device on which the module is (assuming that all the module parameters are on the same device).
        return get_parameter_device(self)

    def set_attention_slice(self, slice_size):
        raise NotImplementedError("Attention slicing is not supported for this model.")

    def is_gradient_checkpointing(self) -> bool:
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())

    def enable_gradient_checkpointing(self):
        self.set_gradient_checkpointing(value=True)

    def disable_gradient_checkpointing(self):
        self.set_gradient_checkpointing(value=False)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool) -> None:
        modules = self.down_blocks + [self.mid_block] + self.up_blocks
        for module in modules:
            module.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool) -> None:
        modules = self.down_blocks + [self.mid_block] + self.up_blocks
        for module in modules:
            module.set_use_sdpa(sdpa)

    def set_gradient_checkpointing(self, value=False):
        modules = self.down_blocks + [self.mid_block] + self.up_blocks
        for module in modules:
            print(module.__class__.__name__, module.gradient_checkpointing, "->", value)
            module.gradient_checkpointing = value

    # endregion

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
    ) -> Union[Dict, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a dict instead of a plain tuple.

        Returns:
            `SampleOutput` or `tuple`:
            `SampleOutput` if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        # デフォルトではサンプルは「2^アップサンプルの数」、つまり64の倍数である必要がある
        # ただそれ以外のサイズにも対応できるように、必要ならアップサンプルのサイズを変更する
        # 多分画質が悪くなるので、64で割り切れるようにしておくのが良い
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        # 64で割り切れないときはupsamplerにサイズを伝える
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        timesteps = self.handle_unusual_timesteps(sample, timesteps)  # 変な時だけ処理

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        # timestepsは重みを含まないので常にfloat32のテンソルを返す
        # しかしtime_embeddingはfp16で動いているかもしれないので、ここでキャストする必要がある
        # time_projでキャストしておけばいいんじゃね？
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            # downblockはforwardで必ずencoder_hidden_statesを受け取るようにしても良さそうだけど、
            # まあこちらのほうがわかりやすいかもしれない
            if downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # skip connectionにControlNetの出力を追加する
        if down_block_additional_residuals is not None:
            down_block_res_samples = list(down_block_res_samples)
            for i in range(len(down_block_res_samples)):
                down_block_res_samples[i] += down_block_additional_residuals[i]
            down_block_res_samples = tuple(down_block_res_samples)

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # ControlNetの出力を追加する
        if mid_block_additional_residual is not None:
            sample += mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]  # skip connection

            # if we have not reached the final block and need to forward the upsample size, we do it here
            # 前述のように最後のブロック以外ではupsample_sizeを伝える
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return SampleOutput(sample=sample)

    def handle_unusual_timesteps(self, sample, timesteps):
        r"""
        timestampsがTensorでない場合、Tensorに変換する。またOnnx/Core MLと互換性のあるようにbatchサイズまでbroadcastする。
        """
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        return timesteps
