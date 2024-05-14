from modules import shared
from modules.sd_hijack_utils import CondFunc

has_ipex = False
try:
    import torch
    import intel_extension_for_pytorch as ipex # noqa: F401
    has_ipex = True
except Exception:
    pass


def check_for_xpu():
    return has_ipex and hasattr(torch, 'xpu') and torch.xpu.is_available()


def get_xpu_device_string():
    if shared.cmd_opts.device_id is not None:
        return f"xpu:{shared.cmd_opts.device_id}"
    return "xpu"


def torch_xpu_gc():
    with torch.xpu.device(get_xpu_device_string()):
        torch.xpu.empty_cache()


has_xpu = check_for_xpu()


# Arc GPU cannot allocate a single block larger than 4GB: https://github.com/intel/compute-runtime/issues/627
# Here we implement a slicing algorithm to split large batch size into smaller chunks,
# so that SDPA of each chunk wouldn't require any allocation larger than ARC_SINGLE_ALLOCATION_LIMIT.
# The heuristic limit (TOTAL_VRAM // 8) is tuned for Intel Arc A770 16G and Arc A750 8G,
# which is the best trade-off between VRAM usage and performance.
ARC_SINGLE_ALLOCATION_LIMIT = {}
orig_sdp_attn_func = torch.nn.functional.scaled_dot_product_attention
def torch_xpu_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *args, **kwargs
):
    # cast to same dtype first
    key = key.to(query.dtype)
    value = value.to(query.dtype)
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(query.dtype)

    N = query.shape[:-2]  # Batch size
    L = query.size(-2)  # Target sequence length
    E = query.size(-1)  # Embedding dimension of the query and key
    S = key.size(-2)  # Source sequence length
    Ev = value.size(-1)  # Embedding dimension of the value

    total_batch_size = torch.numel(torch.empty(N))
    device_id = query.device.index
    if device_id not in ARC_SINGLE_ALLOCATION_LIMIT:
        ARC_SINGLE_ALLOCATION_LIMIT[device_id] = min(torch.xpu.get_device_properties(device_id).total_memory // 8, 4 * 1024 * 1024 * 1024)
    batch_size_limit = max(1, ARC_SINGLE_ALLOCATION_LIMIT[device_id] // (L * S * query.element_size()))

    if total_batch_size <= batch_size_limit:
        return orig_sdp_attn_func(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            *args, **kwargs
        )

    query = torch.reshape(query, (-1, L, E))
    key = torch.reshape(key, (-1, S, E))
    value = torch.reshape(value, (-1, S, Ev))
    if attn_mask is not None:
        attn_mask = attn_mask.view(-1, L, S)
    chunk_count = (total_batch_size + batch_size_limit - 1) // batch_size_limit
    outputs = []
    for i in range(chunk_count):
        attn_mask_chunk = (
            None
            if attn_mask is None
            else attn_mask[i * batch_size_limit : (i + 1) * batch_size_limit, :, :]
        )
        chunk_output = orig_sdp_attn_func(
            query[i * batch_size_limit : (i + 1) * batch_size_limit, :, :],
            key[i * batch_size_limit : (i + 1) * batch_size_limit, :, :],
            value[i * batch_size_limit : (i + 1) * batch_size_limit, :, :],
            attn_mask_chunk,
            dropout_p,
            is_causal,
            *args, **kwargs
        )
        outputs.append(chunk_output)
    result = torch.cat(outputs, dim=0)
    return torch.reshape(result, (*N, L, Ev))


def is_xpu_device(device: str | torch.device = None):
    if device is None:
        return False
    if isinstance(device, str):
        return device.startswith("xpu")
    return device.type == "xpu"


if has_xpu:
    try:
        # torch.Generator supports "xpu" device since 2.1
        torch.Generator("xpu")
    except RuntimeError:
        # W/A for https://github.com/intel/intel-extension-for-pytorch/issues/452: torch.Generator API doesn't support XPU device (for torch < 2.1)
        CondFunc('torch.Generator',
            lambda orig_func, device=None: torch.xpu.Generator(device),
            lambda orig_func, device=None: is_xpu_device(device))

    # W/A for some OPs that could not handle different input dtypes
    CondFunc('torch.nn.functional.layer_norm',
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        orig_func(input.to(weight.data.dtype), normalized_shape, weight, *args, **kwargs),
        lambda orig_func, input, normalized_shape=None, weight=None, *args, **kwargs:
        weight is not None and input.dtype != weight.data.dtype)
    CondFunc('torch.nn.modules.GroupNorm.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.nn.modules.linear.Linear.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.nn.modules.conv.Conv2d.forward',
        lambda orig_func, self, input: orig_func(self, input.to(self.weight.data.dtype)),
        lambda orig_func, self, input: input.dtype != self.weight.data.dtype)
    CondFunc('torch.bmm',
        lambda orig_func, input, mat2, out=None: orig_func(input.to(mat2.dtype), mat2, out=out),
        lambda orig_func, input, mat2, out=None: input.dtype != mat2.dtype)
    CondFunc('torch.cat',
        lambda orig_func, tensors, dim=0, out=None: orig_func([t.to(tensors[0].dtype) for t in tensors], dim=dim, out=out),
        lambda orig_func, tensors, dim=0, out=None: not all(t.dtype == tensors[0].dtype for t in tensors))
    CondFunc('torch.nn.functional.scaled_dot_product_attention',
        lambda orig_func, *args, **kwargs: torch_xpu_scaled_dot_product_attention(*args, **kwargs),
        lambda orig_func, query, *args, **kwargs: query.is_xpu)
