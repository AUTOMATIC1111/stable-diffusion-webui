import torch
import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import

# pylint: disable=protected-access, missing-function-docstring, line-too-long

original_torch_bmm = torch.bmm
def torch_bmm(input, mat2, *, out=None):
    if input.dtype != mat2.dtype:
        mat2 = mat2.to(input.dtype)

    #ARC GPUs can't allocate more than 4GB to a single block, Slice it:
    batch_size_attention, input_tokens, mat2_shape = input.shape[0], input.shape[1], mat2.shape[2]
    block_multiply = 2.4 if input.dtype == torch.float32 else 1.2
    block_size = (batch_size_attention * input_tokens * mat2_shape) / 1024 * block_multiply #MB
    split_slice_size = batch_size_attention
    if block_size >= 4000:
        do_split = True
        #Find something divisible with the input_tokens
        while ((split_slice_size * input_tokens * mat2_shape) / 1024 * block_multiply) > 4000:
            split_slice_size = split_slice_size // 2
            if split_slice_size <= 1:
                split_slice_size = 1
                break
    else:
        do_split = False

    split_block_size = (split_slice_size * input_tokens * mat2_shape) / 1024 * block_multiply #MB
    split_2_slice_size = input_tokens
    if split_block_size >= 4000:
        do_split_2 = True
        #Find something divisible with the input_tokens
        while ((split_slice_size * split_2_slice_size * mat2_shape) / 1024 * block_multiply) > 4000:
            split_2_slice_size = split_2_slice_size // 2
            if split_2_slice_size <= 1:
                split_2_slice_size = 1
                break
    else:
        do_split_2 = False

    if do_split:
        hidden_states = torch.zeros(input.shape[0], input.shape[1], mat2.shape[2], device=input.device, dtype=input.dtype)
        for i in range(batch_size_attention // split_slice_size):
            start_idx = i * split_slice_size
            end_idx = (i + 1) * split_slice_size
            if do_split_2:
                for i2 in range(input_tokens // split_2_slice_size): # pylint: disable=invalid-name
                    start_idx_2 = i2 * split_2_slice_size
                    end_idx_2 = (i2 + 1) * split_2_slice_size
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
    return hidden_states

original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
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
                for i2 in range(query_tokens // split_2_slice_size): # pylint: disable=invalid-name
                    start_idx_2 = i2 * split_2_slice_size
                    end_idx_2 = (i2 + 1) * split_2_slice_size
                    hidden_states[:, start_idx:end_idx, start_idx_2:end_idx_2] = original_scaled_dot_product_attention(
                        query[:, start_idx:end_idx, start_idx_2:end_idx_2],
                        key[:, start_idx:end_idx, start_idx_2:end_idx_2],
                        value[:, start_idx:end_idx, start_idx_2:end_idx_2],
                        attn_mask=attn_mask[:, start_idx:end_idx, start_idx_2:end_idx_2] if attn_mask is not None else attn_mask,
                        dropout_p=dropout_p, is_causal=is_causal
                    )
            else:
                hidden_states[:, start_idx:end_idx] = original_scaled_dot_product_attention(
                    query[:, start_idx:end_idx],
                    key[:, start_idx:end_idx],
                    value[:, start_idx:end_idx],
                    attn_mask=attn_mask[:, start_idx:end_idx] if attn_mask is not None else attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal
                )
    else:
        return original_scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )
    return hidden_states

def attention_init():
    #ARC GPUs can't allocate more than 4GB to a single block:
    torch.bmm = torch_bmm
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
