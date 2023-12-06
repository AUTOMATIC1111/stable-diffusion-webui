# original source:
#   https://github.com/AminRezaei0x443/memory-efficient-attention/blob/1bc0d9e6ac5f82ea43a375135c4e1d3896ee1694/memory_efficient_attention/attention_torch.py
# license:
#   MIT License (see Memory Efficient Attention under the Licenses section in the web UI interface for the full license)
# credit:
#   Amin Rezaei (original author)
#   Alex Birch (optimized algorithm for 3D tensors, at the expense of removing bias, masking and callbacks)
#   brkirch (modified to use torch.narrow instead of dynamic_slice implementation)
# implementation of:
#   Self-attention Does Not Need O(n2) Memory":
#   https://arxiv.org/abs/2112.05682v2

from functools import partial
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, NamedTuple


def narrow_trunc(
    input: Tensor,
    dim: int,
    start: int,
    length: int
) -> Tensor:
    return torch.narrow(input, dim, start, length if input.shape[dim] >= start + length else input.shape[dim] - start)


class AttnChunk(NamedTuple):
    exp_values: Tensor
    exp_weights_sum: Tensor
    max_score: Tensor


class SummarizeChunk:
    @staticmethod
    def __call__(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> AttnChunk: ...


class ComputeQueryChunkAttn:
    @staticmethod
    def __call__(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor: ...


def _summarize_chunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
) -> AttnChunk:
    attn_weights = torch.baddbmm(
        torch.zeros(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key.transpose(1,2),
        alpha=scale,
        beta=0,
    )
    max_score, _ = torch.max(attn_weights, -1, keepdim=True)
    max_score = max_score.detach()
    exp_weights = torch.exp(attn_weights - max_score)
    exp_values = torch.bmm(exp_weights, value) if query.device.type == 'mps' else torch.bmm(exp_weights, value.to(exp_weights.dtype)).to(value.dtype)
    max_score = max_score.squeeze(-1)
    return AttnChunk(exp_values, exp_weights.sum(dim=-1), max_score)


def _query_chunk_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    summarize_chunk: SummarizeChunk,
    kv_chunk_size: int,
) -> Tensor:
    batch_x_heads, k_tokens, k_channels_per_head = key.shape
    _, _, v_channels_per_head = value.shape

    def chunk_scanner(chunk_idx: int) -> AttnChunk:
        key_chunk = narrow_trunc(
            key,
            1,
            chunk_idx,
            kv_chunk_size
        )
        value_chunk = narrow_trunc(
            value,
            1,
            chunk_idx,
            kv_chunk_size
        )
        return summarize_chunk(query, key_chunk, value_chunk)

    chunks: list[AttnChunk] = [
        chunk_scanner(chunk) for chunk in torch.arange(0, k_tokens, kv_chunk_size)
    ]
    acc_chunk = AttnChunk(*map(torch.stack, zip(*chunks)))
    chunk_values, chunk_weights, chunk_max = acc_chunk

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights


# TODO: refactor CrossAttention#get_attention_scores to share code with this
def _get_attention_scores_no_kv_chunking(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
) -> Tensor:
    attn_scores = torch.baddbmm(
        torch.zeros(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key.transpose(1,2),
        alpha=scale,
        beta=0,
    )
    attn_probs = attn_scores.softmax(dim=-1)
    del attn_scores
    hidden_states_slice = torch.bmm(attn_probs, value) if query.device.type == 'mps' else torch.bmm(attn_probs, value.to(attn_probs.dtype)).to(value.dtype)
    return hidden_states_slice


class ScannedChunk(NamedTuple):
    chunk_idx: int
    attn_chunk: AttnChunk


def efficient_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_chunk_size=1024,
    kv_chunk_size: Optional[int] = None,
    kv_chunk_size_min: Optional[int] = None,
    use_checkpoint=True,
):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Args:
        query: queries for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        key: keys for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        value: values to be used in attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        query_chunk_size: int: query chunks size
        kv_chunk_size: Optional[int]: key/value chunks size. if None: defaults to sqrt(key_tokens)
        kv_chunk_size_min: Optional[int]: key/value minimum chunk size. only considered when kv_chunk_size is None. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
        use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
      Returns:
        Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
      """
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, k_tokens, _ = key.shape
    scale = q_channels_per_head ** -0.5

    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)

    def get_query_chunk(chunk_idx: int) -> Tensor:
        return narrow_trunc(
            query,
            1,
            chunk_idx,
            min(query_chunk_size, q_tokens)
        )

    summarize_chunk: SummarizeChunk = partial(_summarize_chunk, scale=scale)
    summarize_chunk: SummarizeChunk = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk
    compute_query_chunk_attn: ComputeQueryChunkAttn = partial(
        _get_attention_scores_no_kv_chunking,
        scale=scale
    ) if k_tokens <= kv_chunk_size else (
        # fast-path for when there's just 1 key-value chunk per query chunk (this is just sliced attention btw)
        partial(
            _query_chunk_attention,
            kv_chunk_size=kv_chunk_size,
            summarize_chunk=summarize_chunk,
        )
    )

    if q_tokens <= query_chunk_size:
        # fast-path for when there's just 1 query chunk
        return compute_query_chunk_attn(
            query=query,
            key=key,
            value=value,
        )

    res = torch.zeros_like(query)
    for i in range(math.ceil(q_tokens / query_chunk_size)):
        attn_scores = compute_query_chunk_attn(
            query=get_query_chunk(i * query_chunk_size),
            key=key,
            value=value,
        )

        res[:, i * query_chunk_size:i * query_chunk_size + attn_scores.shape[1], :] = attn_scores

    return res
