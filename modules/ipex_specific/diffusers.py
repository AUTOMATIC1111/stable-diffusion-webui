import torch
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F
import diffusers #1.19.3

Attention = diffusers.models.attention_processor.Attention

class SlicedAttnProcessor:
    r"""
    Processor for implementing sliced attention.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        batch_size_attention, query_tokens, shape_three = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads), device=query.device, dtype=query.dtype
        )

        #ARC GPUs can't allocate more than 4GB to a single block, Slice it:
        block_multiply = 2.4 if query.dtype == torch.float32 else 1.2
        block_size = (batch_size_attention * query_tokens * shape_three) / 1024 * block_multiply #MB
        split_2_slice_size = query_tokens
        if block_size >= 4000:
            do_split_2 = True
            #Find something divisible with the query_tokens
            while ((self.slice_size * split_2_slice_size * shape_three) / 1024 * block_multiply) > 4000:
                split_2_slice_size = split_2_slice_size // 2
                if split_2_slice_size <= 1:
                    split_2_slice_size = 1
                    break
        else:
            do_split_2 = False

        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            if do_split_2:
                for i2 in range(query_tokens // split_2_slice_size):
                    start_idx_2 = i2 * split_2_slice_size
                    end_idx_2 = (i2 + 1) * split_2_slice_size

                    query_slice = query[start_idx:end_idx, start_idx_2:end_idx_2]
                    key_slice = key[start_idx:end_idx, start_idx_2:end_idx_2]
                    attn_mask_slice = attention_mask[start_idx:end_idx, start_idx_2:end_idx_2] if attention_mask is not None else None

                    attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                    attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx, start_idx_2:end_idx_2])

                    hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = attn_slice
            else:
                query_slice = query[start_idx:end_idx]
                key_slice = key[start_idx:end_idx]
                attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

                attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

                hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        #ARC GPUs can't allocate more than 4GB to a single block, Slice it:
        shape_one, batch_size_attention, query_tokens, shape_four = query.shape
        block_multiply = 2.4 if query.dtype == torch.float32 else 1.2
        block_size = (shape_one * batch_size_attention * query_tokens * shape_four) / 1024 * block_multiply #MB
        split_slice_size = batch_size_attention
        if block_size >= 4000:
            do_split = True
            #Find something divisible with the shape_one
            while ((shape_one * split_slice_size * query_tokens * shape_four) / 1024 * block_multiply) > 4000:
                split_slice_size = split_slice_size // 2
                if split_slice_size <= 1:
                    split_slice_size = 1
                    break
        else:
            do_split = False

        split_block_size = (shape_one * split_slice_size * query_tokens * shape_four) / 1024 * block_multiply #MB
        split_2_slice_size = query_tokens
        if split_block_size >= 4000:
            do_split_2 = True
            #Find something divisible with the batch_size_attention
            while ((shape_one * split_slice_size * split_2_slice_size * shape_four) / 1024 * block_multiply) > 4000:
                split_2_slice_size = split_2_slice_size // 2
                if split_2_slice_size <= 1:
                    split_2_slice_size = 1
                    break
        else:
            do_split_2 = False

        if do_split:
            hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
            for i in range(batch_size_attention // split_slice_size):
                start_idx = i * split_slice_size
                end_idx = (i + 1) * split_slice_size
                if do_split_2:
                    for i2 in range(query_tokens // split_2_slice_size):
                        start_idx_2 = i2 * split_2_slice_size
                        end_idx_2 = (i2 + 1) * split_2_slice_size

                        query_slice = query[:, start_idx:end_idx, start_idx_2:end_idx_2]
                        key_slice = key[:, start_idx:end_idx, start_idx_2:end_idx_2]
                        attn_mask_slice = attention_mask[:, start_idx:end_idx, start_idx_2:end_idx_2] if attention_mask is not None else None

                        attn_slice = F.scaled_dot_product_attention(
                        query_slice, key_slice, value[:, start_idx:end_idx, start_idx_2:end_idx_2],
                        attn_mask=attn_mask_slice, dropout_p=0.0, is_causal=False
                        )
                        hidden_states[:, start_idx:end_idx, start_idx_2:end_idx_2] = attn_slice
                else:
                    query_slice = query[:, start_idx:end_idx]
                    key_slice = key[:, start_idx:end_idx]
                    attn_mask_slice = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None

                    attn_slice = F.scaled_dot_product_attention(
                        query_slice, key_slice, value[:, start_idx:end_idx],
                        attn_mask=attn_mask_slice, dropout_p=0.0, is_causal=False
                    )
                    hidden_states[:, start_idx:end_idx] = attn_slice
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def ipex_diffusers():
    #ARC GPUs can't allocate more than 4GB to a single block:
    diffusers.models.attention_processor.SlicedAttnProcessor = SlicedAttnProcessor
    diffusers.models.attention_processor.AttnProcessor2_0 = AttnProcessor2_0
