import os
import torch
import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
import diffusers #0.24.0 # pylint: disable=import-error
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
from functools import cache

# pylint: disable=protected-access, missing-function-docstring, line-too-long

attention_slice_rate = float(os.environ.get('IPEX_ATTENTION_SLICE_RATE', 4))

@cache
def find_slice_size(slice_size, slice_block_size):
    while (slice_size * slice_block_size) > attention_slice_rate:
        slice_size = slice_size // 2
        if slice_size <= 1:
            slice_size = 1
            break
    return slice_size

@cache
def find_attention_slice_sizes(query_shape, query_element_size, query_device_type, slice_size=None):
    if len(query_shape) == 3:
        batch_size_attention, query_tokens, shape_three = query_shape
        shape_four = 1
    else:
        batch_size_attention, query_tokens, shape_three, shape_four = query_shape
    if slice_size is not None:
        batch_size_attention = slice_size

    slice_block_size = query_tokens * shape_three * shape_four / 1024 / 1024 * query_element_size
    block_size = batch_size_attention * slice_block_size

    split_slice_size = batch_size_attention
    split_2_slice_size = query_tokens
    split_3_slice_size = shape_three

    do_split = False
    do_split_2 = False
    do_split_3 = False

    if query_device_type != "xpu":
        return do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size

    if block_size > attention_slice_rate:
        do_split = True
        split_slice_size = find_slice_size(split_slice_size, slice_block_size)
        if split_slice_size * slice_block_size > attention_slice_rate:
            slice_2_block_size = split_slice_size * shape_three * shape_four / 1024 / 1024 * query_element_size
            do_split_2 = True
            split_2_slice_size = find_slice_size(split_2_slice_size, slice_2_block_size)
            if split_2_slice_size * slice_2_block_size > attention_slice_rate:
                slice_3_block_size = split_slice_size * split_2_slice_size * shape_four / 1024 / 1024 * query_element_size
                do_split_3 = True
                split_3_slice_size = find_slice_size(split_3_slice_size, slice_3_block_size)

    return do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size

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

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor,
    encoder_hidden_states=None, attention_mask=None) -> torch.FloatTensor: # pylint: disable=too-many-statements, too-many-locals, too-many-branches

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

        ####################################################################
        # ARC GPUs can't allocate more than 4GB to a single block, Slice it:
        _, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_attention_slice_sizes(query.shape, query.element_size(), query.device.type, slice_size=self.slice_size)

        for i in range(batch_size_attention // split_slice_size):
            start_idx = i * split_slice_size
            end_idx = (i + 1) * split_slice_size
            if do_split_2:
                for i2 in range(query_tokens // split_2_slice_size): # pylint: disable=invalid-name
                    start_idx_2 = i2 * split_2_slice_size
                    end_idx_2 = (i2 + 1) * split_2_slice_size
                    if do_split_3:
                        for i3 in range(shape_three // split_3_slice_size): # pylint: disable=invalid-name
                            start_idx_3 = i3 * split_3_slice_size
                            end_idx_3 = (i3 + 1) * split_3_slice_size

                            query_slice = query[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3]
                            key_slice = key[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3]
                            attn_mask_slice = attention_mask[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] if attention_mask is not None else None

                            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                            del query_slice
                            del key_slice
                            del attn_mask_slice
                            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3])

                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = attn_slice
                            del attn_slice
                    else:
                        query_slice = query[start_idx:end_idx, start_idx_2:end_idx_2]
                        key_slice = key[start_idx:end_idx, start_idx_2:end_idx_2]
                        attn_mask_slice = attention_mask[start_idx:end_idx, start_idx_2:end_idx_2] if attention_mask is not None else None

                        attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                        del query_slice
                        del key_slice
                        del attn_mask_slice
                        attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx, start_idx_2:end_idx_2])

                        hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = attn_slice
                        del attn_slice
                torch.xpu.synchronize(query.device)
            else:
                query_slice = query[start_idx:end_idx]
                key_slice = key[start_idx:end_idx]
                attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

                attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                del query_slice
                del key_slice
                del attn_mask_slice
                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

                hidden_states[start_idx:end_idx] = attn_slice
                del attn_slice
        ####################################################################

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


class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor,
    encoder_hidden_states=None, attention_mask=None,
    temb=None, scale: float = 1.0) -> torch.Tensor: # pylint: disable=too-many-statements, too-many-locals, too-many-branches

        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

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

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        ####################################################################
        # ARC GPUs can't allocate more than 4GB to a single block, Slice it:
        batch_size_attention, query_tokens, shape_three = query.shape[0], query.shape[1], query.shape[2]
        hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
        do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_attention_slice_sizes(query.shape, query.element_size(), query.device.type)

        if do_split:
            for i in range(batch_size_attention // split_slice_size):
                start_idx = i * split_slice_size
                end_idx = (i + 1) * split_slice_size
                if do_split_2:
                    for i2 in range(query_tokens // split_2_slice_size): # pylint: disable=invalid-name
                        start_idx_2 = i2 * split_2_slice_size
                        end_idx_2 = (i2 + 1) * split_2_slice_size
                        if do_split_3:
                            for i3 in range(shape_three // split_3_slice_size): # pylint: disable=invalid-name
                                start_idx_3 = i3 * split_3_slice_size
                                end_idx_3 = (i3 + 1) * split_3_slice_size

                                query_slice = query[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3]
                                key_slice = key[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3]
                                attn_mask_slice = attention_mask[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] if attention_mask is not None else None

                                attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                                del query_slice
                                del key_slice
                                del attn_mask_slice
                                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3])

                                hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = attn_slice
                                del attn_slice
                        else:
                            query_slice = query[start_idx:end_idx, start_idx_2:end_idx_2]
                            key_slice = key[start_idx:end_idx, start_idx_2:end_idx_2]
                            attn_mask_slice = attention_mask[start_idx:end_idx, start_idx_2:end_idx_2] if attention_mask is not None else None

                            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                            del query_slice
                            del key_slice
                            del attn_mask_slice
                            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx, start_idx_2:end_idx_2])

                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = attn_slice
                            del attn_slice
                else:
                    query_slice = query[start_idx:end_idx]
                    key_slice = key[start_idx:end_idx]
                    attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

                    attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                    del query_slice
                    del key_slice
                    del attn_mask_slice
                    attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

                    hidden_states[start_idx:end_idx] = attn_slice
                    del attn_slice
            torch.xpu.synchronize(query.device)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
        ####################################################################
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
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
    diffusers.models.attention_processor.AttnProcessor = AttnProcessor
