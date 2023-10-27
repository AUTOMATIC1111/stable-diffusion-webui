# Diffusersのコードをベースとした sd_xl_baseのU-Net
# state dictの形式をSDXLに合わせてある

"""
      target: sgm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        adm_in_channels: 2816
        num_classes: sequential
        use_checkpoint: True
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 4]
        num_head_channels: 64
        use_spatial_transformer: True
        use_linear_in_transformer: True
        transformer_depth: [1, 2, 10]  # note: the first is unused (due to attn_res starting at 2) 32, 16, 8 --> 64, 32, 16
        context_dim: 2048
        spatial_transformer_attn_type: softmax-xformers
        legacy: False
"""

import math
from types import SimpleNamespace
from typing import Optional
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from einops import rearrange


IN_CHANNELS: int = 4
OUT_CHANNELS: int = 4
ADM_IN_CHANNELS: int = 2816
CONTEXT_DIM: int = 2048
MODEL_CHANNELS: int = 320
TIME_EMBED_DIM = 320 * 4

USE_REENTRANT = True

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

    # concat sine and cosine embeddings: flipped from Diffusers original ver because always flip_sin_to_cos=True
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        if self.weight.dtype != torch.float32:
            return super().forward(x)
        return super().forward(x.float()).type(x.dtype)


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(TIME_EMBED_DIM, out_channels))

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Identity(),  # to make state_dict compatible with original model
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip_connection = nn.Identity()

        self.gradient_checkpointing = False

    def forward_body(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        h = h + emb_out[:, :, None, None]
        h = self.out_layers(h)
        x = self.skip_connection(x)
        return x + h

    def forward(self, x, emb):
        if self.training and self.gradient_checkpointing:
            # print("ResnetBlock2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.forward_body), x, emb, use_reentrant=USE_REENTRANT)
        else:
            x = self.forward_body(x, emb)

        return x


class Downsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=1)

        self.gradient_checkpointing = False

    def forward_body(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.op(hidden_states)

        return hidden_states

    def forward(self, hidden_states):
        if self.training and self.gradient_checkpointing:
            # print("Downsample2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.forward_body), hidden_states, use_reentrant=USE_REENTRANT
            )
        else:
            hidden_states = self.forward_body(hidden_states)

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
        del q, k, v

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

        self.gradient_checkpointing = False

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

    def forward_body(self, hidden_states, context=None, timestep=None):
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        hidden_states = self.attn1(norm_hidden_states) + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(norm_hidden_states, context=context) + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states

    def forward(self, hidden_states, context=None, timestep=None):
        if self.training and self.gradient_checkpointing:
            # print("BasicTransformerBlock: checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.forward_body), hidden_states, context, timestep, use_reentrant=USE_REENTRANT
            )
        else:
            output = self.forward_body(hidden_states, context, timestep)

        return output


class Transformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.use_linear_projection = use_linear_projection

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        # self.norm = GroupNorm32(32, in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        blocks = []
        for _ in range(num_transformer_layers):
            blocks.append(
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                )
            )

        self.transformer_blocks = nn.ModuleList(blocks)

        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.gradient_checkpointing = False

    def set_use_memory_efficient_attention(self, xformers, mem_eff):
        for transformer in self.transformer_blocks:
            transformer.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa):
        for transformer in self.transformer_blocks:
            transformer.set_use_sdpa(sdpa)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None):
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

        return output


class Upsample2D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward_body(self, hidden_states, output_size=None):
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

    def forward(self, hidden_states, output_size=None):
        if self.training and self.gradient_checkpointing:
            # print("Upsample2D: gradient_checkpointing")

            def create_custom_forward(func):
                def custom_forward(*inputs):
                    return func(*inputs)

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.forward_body), hidden_states, output_size, use_reentrant=USE_REENTRANT
            )
        else:
            hidden_states = self.forward_body(hidden_states, output_size)

        return hidden_states


class SdxlUNet2DConditionModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = IN_CHANNELS
        self.out_channels = OUT_CHANNELS
        self.model_channels = MODEL_CHANNELS
        self.time_embed_dim = TIME_EMBED_DIM
        self.adm_in_channels = ADM_IN_CHANNELS

        self.gradient_checkpointing = False
        # self.sample_size = sample_size

        # time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # label embedding
        self.label_emb = nn.Sequential(
            nn.Sequential(
                nn.Linear(self.adm_in_channels, self.time_embed_dim),
                nn.SiLU(),
                nn.Linear(self.time_embed_dim, self.time_embed_dim),
            )
        )

        # input
        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=(1, 1)),
                )
            ]
        )

        # level 0
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=1 * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            ]
            self.input_blocks.append(nn.ModuleList(layers))

        self.input_blocks.append(
            nn.Sequential(
                Downsample2D(
                    channels=1 * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            )
        )

        # level 1
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=(1 if i == 0 else 2) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            self.input_blocks.append(nn.ModuleList(layers))

        self.input_blocks.append(
            nn.Sequential(
                Downsample2D(
                    channels=2 * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
            )
        )

        # level 2
        for i in range(2):
            layers = [
                ResnetBlock2D(
                    in_channels=(2 if i == 0 else 4) * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            self.input_blocks.append(nn.ModuleList(layers))

        # mid
        self.middle_block = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels=4 * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
                ResnetBlock2D(
                    in_channels=4 * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
            ]
        )

        # output
        self.output_blocks = nn.ModuleList([])

        # level 2
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=4 * self.model_channels + (4 if i <= 1 else 2) * self.model_channels,
                    out_channels=4 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=4 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=4 * self.model_channels,
                    num_transformer_layers=10,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            if i == 2:
                layers.append(
                    Upsample2D(
                        channels=4 * self.model_channels,
                        out_channels=4 * self.model_channels,
                    )
                )

            self.output_blocks.append(nn.ModuleList(layers))

        # level 1
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=2 * self.model_channels + (4 if i == 0 else (2 if i == 1 else 1)) * self.model_channels,
                    out_channels=2 * self.model_channels,
                ),
                Transformer2DModel(
                    num_attention_heads=2 * self.model_channels // 64,
                    attention_head_dim=64,
                    in_channels=2 * self.model_channels,
                    num_transformer_layers=2,
                    use_linear_projection=True,
                    cross_attention_dim=2048,
                ),
            ]
            if i == 2:
                layers.append(
                    Upsample2D(
                        channels=2 * self.model_channels,
                        out_channels=2 * self.model_channels,
                    )
                )

            self.output_blocks.append(nn.ModuleList(layers))

        # level 0
        for i in range(3):
            layers = [
                ResnetBlock2D(
                    in_channels=1 * self.model_channels + (2 if i == 0 else 1) * self.model_channels,
                    out_channels=1 * self.model_channels,
                ),
            ]

            self.output_blocks.append(nn.ModuleList(layers))

        # output
        self.out = nn.ModuleList(
            [GroupNorm32(32, self.model_channels), nn.SiLU(), nn.Conv2d(self.model_channels, self.out_channels, 3, padding=1)]
        )

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
        self.gradient_checkpointing = True
        self.set_gradient_checkpointing(value=True)

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.set_gradient_checkpointing(value=False)

    def set_use_memory_efficient_attention(self, xformers: bool, mem_eff: bool) -> None:
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block:
                if hasattr(module, "set_use_memory_efficient_attention"):
                    # print(module.__class__.__name__)
                    module.set_use_memory_efficient_attention(xformers, mem_eff)

    def set_use_sdpa(self, sdpa: bool) -> None:
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block:
                if hasattr(module, "set_use_sdpa"):
                    module.set_use_sdpa(sdpa)

    def set_gradient_checkpointing(self, value=False):
        blocks = self.input_blocks + [self.middle_block] + self.output_blocks
        for block in blocks:
            for module in block.modules():
                if hasattr(module, "gradient_checkpointing"):
                    # print(module.__class__.__name__, module.gradient_checkpointing, "->", value)
                    module.gradient_checkpointing = value

    # endregion

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # broadcast timesteps to batch dimension
        timesteps = timesteps.expand(x.shape[0])

        hs = []
        t_emb = get_timestep_embedding(timesteps, self.model_channels)  # , repeat_only=False)
        t_emb = t_emb.to(x.dtype)
        emb = self.time_embed(t_emb)

        assert x.shape[0] == y.shape[0], f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
        assert x.dtype == y.dtype, f"dtype mismatch: {x.dtype} != {y.dtype}"
        # assert x.dtype == self.dtype
        emb = emb + self.label_emb(y)

        def call_module(module, h, emb, context):
            x = h
            for layer in module:
                # print(layer.__class__.__name__, x.dtype, emb.dtype, context.dtype if context is not None else None)
                if isinstance(layer, ResnetBlock2D):
                    x = layer(x, emb)
                elif isinstance(layer, Transformer2DModel):
                    x = layer(x, context)
                else:
                    x = layer(x)
            return x

        # h = x.type(self.dtype)
        h = x
        for module in self.input_blocks:
            h = call_module(module, h, emb, context)
            hs.append(h)

        h = call_module(self.middle_block, h, emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = call_module(module, h, emb, context)

        h = h.type(x.dtype)
        h = call_module(self.out, h, emb, context)

        return h


if __name__ == "__main__":
    import time

    print("create unet")
    unet = SdxlUNet2DConditionModel()

    unet.to("cuda")
    unet.set_use_memory_efficient_attention(True, False)
    unet.set_gradient_checkpointing(True)
    unet.train()

    # 使用メモリ量確認用の疑似学習ループ
    print("preparing optimizer")

    # optimizer = torch.optim.SGD(unet.parameters(), lr=1e-3, nesterov=True, momentum=0.9) # not working

    # import bitsandbytes
    # optimizer = bitsandbytes.adam.Adam8bit(unet.parameters(), lr=1e-3)        # not working
    # optimizer = bitsandbytes.optim.RMSprop8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2
    # optimizer=bitsandbytes.optim.Adagrad8bit(unet.parameters(), lr=1e-3)  # working at 23.5 GB with torch2

    import transformers

    optimizer = transformers.optimization.Adafactor(unet.parameters(), relative_step=True)  # working at 22.2GB with torch2

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("start training")
    steps = 10
    batch_size = 1

    for step in range(steps):
        print(f"step {step}")
        if step == 1:
            time_start = time.perf_counter()

        x = torch.randn(batch_size, 4, 128, 128).cuda()  # 1024x1024
        t = torch.randint(low=0, high=10, size=(batch_size,), device="cuda")
        ctx = torch.randn(batch_size, 77, 2048).cuda()
        y = torch.randn(batch_size, ADM_IN_CHANNELS).cuda()

        with torch.cuda.amp.autocast(enabled=True):
            output = unet(x, t, ctx, y)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    time_end = time.perf_counter()
    print(f"elapsed time: {time_end - time_start} [sec] for last {steps - 1} steps")
