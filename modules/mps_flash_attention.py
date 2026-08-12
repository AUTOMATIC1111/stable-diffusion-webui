"""Selective Draw Things-style Metal Flash Attention for Apple Silicon."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys

import torch
import torch.nn.functional as F


MIN_TORCH_VERSION = (2, 3)

_extension = None
_availability = None
_availability_error = None
_dispatch_count = 0
_fallback_count = 0
_runtime_failure_warned = False
_first_dispatch_logged = False
_native_sdpa_supports_gqa = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")


def _version_tuple(version):
    components = version.split("+", 1)[0].split(".")
    try:
        return tuple(int(component) for component in components[:2])
    except ValueError:
        return (0, 0)


def should_use_mfa_shape(query_tokens, key_tokens, head_dim):
    """Return whether the measured M1 routing table favors MFA for this shape."""
    if query_tokens < 192:
        return False

    # These are the SD 1.x UNet head dimensions. With MFA encoded on the
    # current command buffer, measured self- and cross-attention shapes all
    # beat PyTorch SDPA without introducing a submission per attention call.
    return head_dim in (40, 80, 160)


def _run_isolated_self_test():
    code = """
import torch
import torch.nn.functional as F
from metal_flash_sdpa import MetalFlashAttentionForward, fused_geglu_forward, fused_group_norm_silu_add_embedding_forward, fused_group_norm_silu_forward

torch.manual_seed(1)
source = torch.randn((1, 256, 320), device='mps', dtype=torch.float16)
q = source.view(1, 256, 8, 40).transpose(1, 2)
k = q.clone()
v = q.clone()
projection = torch.randn((320, 320), device='mps', dtype=torch.float16)
expected = F.scaled_dot_product_attention(q, k, v)
expected = F.linear(expected.transpose(1, 2).reshape(1, 256, 320), projection)
torch.mps.synchronize()
actual = MetalFlashAttentionForward.apply(q, k, v, 40 ** -0.5, False)
actual = F.linear(actual.transpose(1, 2).reshape(1, 256, 320), projection)
torch.mps.synchronize()
assert torch.isfinite(actual).all().item()
assert (actual.float() - expected.float()).abs().max().item() < 0.05

norm_source = torch.randn((1, 320, 48, 80), device='mps', dtype=torch.float16)
norm_weight = torch.randn((320,), device='mps', dtype=torch.float16)
norm_bias = torch.randn((320,), device='mps', dtype=torch.float16)
expected_norm = F.silu(F.group_norm(norm_source, 32, norm_weight, norm_bias, 1e-5))
actual_norm = fused_group_norm_silu_forward(norm_source, norm_weight, norm_bias, 32, 1e-5)
actual_norm = actual_norm + 0
torch.mps.synchronize()
norm_difference = (actual_norm.float() - expected_norm.float()).abs()
assert torch.isfinite(actual_norm).all().item()
assert norm_difference.max().item() < 0.02
assert norm_difference.mean().item() < 0.001

embedding = torch.randn((1, 320), device='mps', dtype=torch.float16)
expected_embedding_norm = F.silu(F.group_norm(norm_source + embedding[:, :, None, None], 32, norm_weight, norm_bias, 1e-5))
actual_embedding_norm = fused_group_norm_silu_add_embedding_forward(norm_source, embedding, norm_weight, norm_bias, 32, 1e-5) + 0
torch.mps.synchronize()
embedding_difference = (actual_embedding_norm.float() - expected_embedding_norm.float()).abs()
assert torch.isfinite(actual_embedding_norm).all().item()
assert embedding_difference.max().item() < 0.02
assert embedding_difference.mean().item() < 0.001

import numpy as np
geglu_source = torch.randn((1, 256, 2560), device='mps', dtype=torch.float16)
half_values = np.arange(65536, dtype=np.uint16).view(np.float16).copy()
geglu_lut = F.gelu(torch.from_numpy(half_values).to('mps')).contiguous()
value, gate = geglu_source.chunk(2, dim=-1)
expected_geglu = value * F.gelu(gate)
actual_geglu = fused_geglu_forward(geglu_source, geglu_lut) + 0
torch.mps.synchronize()
assert torch.isfinite(actual_geglu).all().item()
assert torch.equal(actual_geglu, expected_geglu)
"""
    environment = os.environ.copy()
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=90,
        env=environment,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"native self-test exited {result.returncode}")


def is_available():
    """Load and crash-test the optional native extension once."""
    global _extension, _availability, _availability_error
    if _availability is not None:
        return _availability

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        _availability = False
        return False
    if not torch.backends.mps.is_available():
        _availability = False
        return False
    if _version_tuple(torch.__version__) < MIN_TORCH_VERSION:
        _availability_error = f"requires PyTorch {MIN_TORCH_VERSION[0]}.{MIN_TORCH_VERSION[1]} or newer"
        _availability = False
        return False

    try:
        _extension = importlib.import_module("metal_flash_sdpa")
        if not hasattr(_extension, "MetalFlashAttentionForward"):
            raise RuntimeError("extension does not expose MetalFlashAttentionForward")
        if _version_tuple(torch.__version__) < (2, 11) and not getattr(_extension, "A1111_MPS_DEFERRED_COMMIT", False):
            raise RuntimeError("native extension is missing the A1111 deferred MPS commit patch")
        if not getattr(_extension, "A1111_MPS_FUSED_GROUP_NORM_SILU", False):
            raise RuntimeError("native extension is missing fused GroupNorm+SiLU")
        if not getattr(_extension, "A1111_MPS_FUSED_GROUP_NORM_SILU_EMBEDDING", False):
            raise RuntimeError("native extension is missing fused GroupNorm+SiLU+embedding")
        if not getattr(_extension, "A1111_MPS_FUSED_GEGLU", False):
            raise RuntimeError("native extension is missing fused GEGLU")
        _run_isolated_self_test()
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        _availability_error = str(exc)
        _availability = False
        print(f"Metal Flash Attention unavailable: {_availability_error}")
        return False

    _availability = True
    print("Metal self-test passed; deferred MFA, fused GroupNorm+SiLU, and fused GEGLU routing enabled.")
    return True


def _can_dispatch(query, key, value, attn_mask, dropout_p, enable_gqa, training):
    if not is_available() or training or enable_gqa or dropout_p != 0.0 or attn_mask is not None:
        return False
    if query.device.type != "mps" or key.device != query.device or value.device != query.device:
        return False
    if query.dtype != torch.float16 or key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if query.shape[0] != key.shape[0] or key.shape != value.shape:
        return False
    if query.shape[1] != key.shape[1] or query.shape[-1] != key.shape[-1]:
        return False
    if torch.is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad):
        return False
    return should_use_mfa_shape(query.shape[-2], key.shape[-2], query.shape[-1])


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    *,
    training=False,
):
    """Use native MFA for measured winning shapes, otherwise use PyTorch SDPA."""
    global _dispatch_count, _fallback_count, _runtime_failure_warned, _first_dispatch_logged
    if _can_dispatch(query, key, value, attn_mask, dropout_p, enable_gqa, training):
        try:
            _dispatch_count += 1
            if not _first_dispatch_logged:
                print(f"Metal Flash Attention first dispatch: Q={tuple(query.shape)}, K={tuple(key.shape)}")
                _first_dispatch_logged = True
            attention_scale = scale if scale is not None else query.shape[-1] ** -0.5
            return _extension.MetalFlashAttentionForward.apply(query, key, value, attention_scale, is_causal)
        except RuntimeError as exc:
            if not _runtime_failure_warned:
                print(f"Metal Flash Attention failed; using PyTorch SDPA: {exc}")
                _runtime_failure_warned = True
            torch.mps.empty_cache()

    _fallback_count += 1
    native_kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
    }
    if enable_gqa:
        if not _native_sdpa_supports_gqa:
            raise RuntimeError("grouped-query attention requires a newer PyTorch SDPA runtime")
        native_kwargs["enable_gqa"] = True

    return F.scaled_dot_product_attention(query, key, value, **native_kwargs)


def diagnostics():
    return {
        "available": bool(_availability),
        "error": _availability_error,
        "dispatches": _dispatch_count,
        "fallbacks": _fallback_count,
    }
