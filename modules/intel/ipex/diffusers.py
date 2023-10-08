import torch
import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
import diffusers #0.21.1 # pylint: disable=import-error
from diffusers.models.attention_processor import Attention

# pylint: disable=protected-access, missing-function-docstring, line-too-long

class SlicedAttnProcessor: # pylint: disable=too-few-public-methods
    r"""
    Processor for implementing sliced attention.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None): # pylint: disable=too-many-statements, too-many-locals, too-many-branches
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
        block_multiply = query.element_size()
        slice_block_size = self.slice_size * shape_three / 1024 / 1024 * block_multiply
        block_size = query_tokens * slice_block_size
        split_2_slice_size = query_tokens
        if block_size > 4:
            do_split_2 = True
            #Find something divisible with the query_tokens
            while (split_2_slice_size * slice_block_size) > 4:
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
                for i2 in range(query_tokens // split_2_slice_size): # pylint: disable=invalid-name
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

def ipex_diffusers():
    #ARC GPUs can't allocate more than 4GB to a single block:
    diffusers.models.attention_processor.SlicedAttnProcessor = SlicedAttnProcessor
