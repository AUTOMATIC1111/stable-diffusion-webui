import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    SlicedAttnProcessor,
    XFormersAttnProcessor
)

try:
    import xformers.ops
except:
    xformers = None


loaded_networks = []


def apply_single_hypernetwork(
    hypernetwork, hidden_states, encoder_hidden_states
):
    context_k, context_v = hypernetwork.forward(hidden_states, encoder_hidden_states)
    return context_k, context_v


def apply_hypernetworks(context_k, context_v, layer=None):
    if len(loaded_networks) == 0:
        return context_v, context_v
    for hypernetwork in loaded_networks:
        context_k, context_v = hypernetwork.forward(context_k, context_v)

    context_k = context_k.to(dtype=context_k.dtype)
    context_v = context_v.to(dtype=context_k.dtype)

    return context_k, context_v



def xformers_forward(
    self: XFormersAttnProcessor,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
):
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    context_k, context_v = apply_hypernetworks(hidden_states, encoder_hidden_states)

    key = attn.to_k(context_k)
    value = attn.to_v(context_v)

    query = attn.head_to_batch_dim(query).contiguous()
    key = attn.head_to_batch_dim(key).contiguous()
    value = attn.head_to_batch_dim(value).contiguous()

    hidden_states = xformers.ops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attention_mask,
        op=self.attention_op,
        scale=attn.scale,
    )
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def sliced_attn_forward(
    self: SlicedAttnProcessor,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
):
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    query = attn.to_q(hidden_states)
    dim = query.shape[-1]
    query = attn.head_to_batch_dim(query)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    context_k, context_v = apply_hypernetworks(hidden_states, encoder_hidden_states)

    key = attn.to_k(context_k)
    value = attn.to_v(context_v)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    batch_size_attention, query_tokens, _ = query.shape
    hidden_states = torch.zeros(
        (batch_size_attention, query_tokens, dim // attn.heads),
        device=query.device,
        dtype=query.dtype,
    )

    for i in range(batch_size_attention // self.slice_size):
        start_idx = i * self.slice_size
        end_idx = (i + 1) * self.slice_size

        query_slice = query[start_idx:end_idx]
        key_slice = key[start_idx:end_idx]
        attn_mask_slice = (
            attention_mask[start_idx:end_idx] if attention_mask is not None else None
        )

        attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

        attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

        hidden_states[start_idx:end_idx] = attn_slice

    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


def v2_0_forward(
    self: AttnProcessor2_0,
    attn: Attention,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
):
    batch_size, sequence_length, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    inner_dim = hidden_states.shape[-1]

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(
            batch_size, attn.heads, -1, attention_mask.shape[-1]
        )

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    context_k, context_v = apply_hypernetworks(hidden_states, encoder_hidden_states)

    key = attn.to_k(context_k)
    value = attn.to_v(context_v)

    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


def replace_attentions_for_hypernetwork():
    import diffusers.models.attention_processor

    diffusers.models.attention_processor.XFormersAttnProcessor.__call__ = (
        xformers_forward
    )
    diffusers.models.attention_processor.SlicedAttnProcessor.__call__ = (
        sliced_attn_forward
    )
    diffusers.models.attention_processor.AttnProcessor2_0.__call__ = v2_0_forward
