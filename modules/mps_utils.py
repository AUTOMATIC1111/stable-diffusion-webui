import psutil


def should_use_sdp(batch, heads, query_tokens, key_tokens, element_size, total_memory=None, available_memory=None):
    """Use native SDPA only while its intermediates fit unified memory safely."""
    if total_memory is None or available_memory is None:
        memory = psutil.virtual_memory()
        total_memory = memory.total
        available_memory = memory.available

    attention_bytes = batch * heads * query_tokens * key_tokens * element_size
    estimated_peak = int(attention_bytes * 2.5)
    budget = min(
        int(total_memory * 0.10),
        int(available_memory * 0.20),
        1536 * 1024 * 1024,
    )
    return estimated_peak <= budget


def attention_query_chunk_size(requested_size, batch_heads, key_tokens, element_size, total_memory=None, available_memory=None):
    """Bound an MPS query tile so attention intermediates stay memory-safe."""
    if total_memory is None or available_memory is None:
        memory = psutil.virtual_memory()
        total_memory = memory.total
        available_memory = memory.available

    peak_budget = min(
        int(total_memory * 0.025),
        int(available_memory * 0.10),
        384 * 1024 * 1024,
    )
    bytes_per_query = max(int(batch_heads * key_tokens * element_size * 2.5), 1)
    maximum_size = max(peak_budget // bytes_per_query, 1)
    if maximum_size >= 64:
        maximum_size = (maximum_size // 64) * 64
    return min(requested_size, maximum_size)
