import os
import torch
import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
from functools import cache

# pylint: disable=protected-access, missing-function-docstring, line-too-long

# ARC GPUs can't allocate more than 4GB to a single block so we slice the attetion layers

sdpa_slice_trigger_rate = float(os.environ.get('IPEX_SDPA_SLICE_TRIGGER_RATE', 6))
attention_slice_rate = float(os.environ.get('IPEX_ATTENTION_SLICE_RATE', 4))

# Find something divisible with the input_tokens
@cache
def find_slice_size(slice_size, slice_block_size):
    while (slice_size * slice_block_size) > attention_slice_rate:
        slice_size = slice_size // 2
        if slice_size <= 1:
            slice_size = 1
            break
    return slice_size

# Find slice sizes for SDPA
@cache
def find_sdpa_slice_sizes(query_shape, query_element_size):
    if len(query_shape) == 3:
        batch_size_attention, query_tokens, shape_three = query_shape
        shape_four = 1
    else:
        batch_size_attention, query_tokens, shape_three, shape_four = query_shape

    slice_block_size = query_tokens * shape_three * shape_four / 1024 / 1024 * query_element_size
    block_size = batch_size_attention * slice_block_size

    split_slice_size = batch_size_attention
    split_2_slice_size = query_tokens
    split_3_slice_size = shape_three

    do_split = False
    do_split_2 = False
    do_split_3 = False

    if block_size > sdpa_slice_trigger_rate:
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

# Find slice sizes for BMM
@cache
def find_bmm_slice_sizes(input_shape, input_element_size, mat2_shape):
    batch_size_attention, input_tokens, mat2_atten_shape = input_shape[0], input_shape[1], mat2_shape[2]
    slice_block_size = input_tokens * mat2_atten_shape / 1024 / 1024 * input_element_size
    block_size = batch_size_attention * slice_block_size

    split_slice_size = batch_size_attention
    split_2_slice_size = input_tokens
    split_3_slice_size = mat2_atten_shape

    do_split = False
    do_split_2 = False
    do_split_3 = False

    if block_size > attention_slice_rate:
        do_split = True
        split_slice_size = find_slice_size(split_slice_size, slice_block_size)
        if split_slice_size * slice_block_size > attention_slice_rate:
            slice_2_block_size = split_slice_size * mat2_atten_shape / 1024 / 1024 * input_element_size
            do_split_2 = True
            split_2_slice_size = find_slice_size(split_2_slice_size, slice_2_block_size)
            if split_2_slice_size * slice_2_block_size > attention_slice_rate:
                slice_3_block_size = split_slice_size * split_2_slice_size / 1024 / 1024 * input_element_size
                do_split_3 = True
                split_3_slice_size = find_slice_size(split_3_slice_size, slice_3_block_size)

    return do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size


original_torch_bmm = torch.bmm
def torch_bmm_32_bit(input, mat2, *, out=None):
    if input.device.type != "xpu":
        return original_torch_bmm(input, mat2, out=out)
    do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_bmm_slice_sizes(input.shape, input.element_size(), mat2.shape)

    # Slice BMM
    if do_split:
        batch_size_attention, input_tokens, mat2_atten_shape = input.shape[0], input.shape[1], mat2.shape[2]
        hidden_states = torch.zeros(input.shape[0], input.shape[1], mat2.shape[2], device=input.device, dtype=input.dtype)
        for i in range(batch_size_attention // split_slice_size):
            start_idx = i * split_slice_size
            end_idx = (i + 1) * split_slice_size
            if do_split_2:
                for i2 in range(input_tokens // split_2_slice_size): # pylint: disable=invalid-name
                    start_idx_2 = i2 * split_2_slice_size
                    end_idx_2 = (i2 + 1) * split_2_slice_size
                    if do_split_3:
                        for i3 in range(mat2_atten_shape // split_3_slice_size): # pylint: disable=invalid-name
                            start_idx_3 = i3 * split_3_slice_size
                            end_idx_3 = (i3 + 1) * split_3_slice_size
                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = original_torch_bmm(
                                input[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                mat2[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                out=out
                            )
                    else:
                        hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = original_torch_bmm(
                            input[start_idx:end_idx, start_idx_2:end_idx_2],
                            mat2[start_idx:end_idx, start_idx_2:end_idx_2],
                            out=out
                        )
            else:
                hidden_states[start_idx:end_idx] = original_torch_bmm(
                    input[start_idx:end_idx],
                    mat2[start_idx:end_idx],
                    out=out
                )
    else:
        return original_torch_bmm(input, mat2, out=out)
    torch.xpu.synchronize(input.device)
    return hidden_states

original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
def scaled_dot_product_attention_32_bit(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    if query.device.type != "xpu":
        return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_sdpa_slice_sizes(query.shape, query.element_size())

    # Slice SDPA
    if do_split:
        batch_size_attention, query_tokens, shape_three = query.shape[0], query.shape[1], query.shape[2]
        hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
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
                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = original_scaled_dot_product_attention(
                                query[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                key[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                value[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                attn_mask=attn_mask[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] if attn_mask is not None else attn_mask,
                                dropout_p=dropout_p, is_causal=is_causal
                            )
                    else:
                        hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = original_scaled_dot_product_attention(
                            query[start_idx:end_idx, start_idx_2:end_idx_2],
                            key[start_idx:end_idx, start_idx_2:end_idx_2],
                            value[start_idx:end_idx, start_idx_2:end_idx_2],
                            attn_mask=attn_mask[start_idx:end_idx, start_idx_2:end_idx_2] if attn_mask is not None else attn_mask,
                            dropout_p=dropout_p, is_causal=is_causal
                        )
            else:
                hidden_states[start_idx:end_idx] = original_scaled_dot_product_attention(
                    query[start_idx:end_idx],
                    key[start_idx:end_idx],
                    value[start_idx:end_idx],
                    attn_mask=attn_mask[start_idx:end_idx] if attn_mask is not None else attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal
                )
    else:
        return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    torch.xpu.synchronize(query.device)
    return hidden_states
