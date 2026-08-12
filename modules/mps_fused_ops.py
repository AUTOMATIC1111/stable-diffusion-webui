"""Measured native Metal fusions for Apple Silicon inference."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from modules import mps_flash_attention


_dispatch_count = 0
_fallback_count = 0
_runtime_failure_warned = False
_first_dispatch_logged = False
_runtime_disabled = False
_geglu_dispatch_count = 0
_geglu_fallback_count = 0
_geglu_runtime_failure_warned = False
_geglu_first_dispatch_logged = False
_geglu_runtime_disabled = False
_geglu_lut = None


def _can_dispatch(input_tensor, norm):
    if _runtime_disabled:
        return False
    if os.environ.get("A1111_MPS_DISABLE_FUSED_GROUP_NORM_SILU") == "1":
        return False
    from modules import shared

    if not getattr(shared.opts, "mps_fused_group_norm_silu", True):
        return False
    if input_tensor.device.type != "mps" or input_tensor.dtype != torch.float16:
        return False
    if input_tensor.ndim != 4 or not input_tensor.is_contiguous():
        return False
    if norm.weight is None or norm.bias is None:
        return False
    if norm.weight.device != input_tensor.device or norm.bias.device != input_tensor.device:
        return False
    if norm.weight.dtype != input_tensor.dtype or norm.bias.dtype != input_tensor.dtype:
        return False
    if not norm.weight.is_contiguous() or not norm.bias.is_contiguous():
        return False
    if input_tensor.shape[1] % norm.num_groups != 0:
        return False
    if torch.is_grad_enabled() and (
        input_tensor.requires_grad or norm.weight.requires_grad or norm.bias.requires_grad
    ):
        return False
    return mps_flash_attention.is_available()


def group_norm_silu(input_tensor, norm):
    """Fuse GroupNorm and SiLU when the measured MPS inference path supports it."""
    global _dispatch_count, _fallback_count, _runtime_failure_warned
    global _first_dispatch_logged, _runtime_disabled
    if _can_dispatch(input_tensor, norm):
        try:
            result = mps_flash_attention._extension.fused_group_norm_silu_forward(
                input_tensor,
                norm.weight,
                norm.bias,
                norm.num_groups,
                norm.eps,
            )
            _dispatch_count += 1
            if not _first_dispatch_logged:
                print(f"Fused Metal GroupNorm+SiLU first dispatch: {tuple(input_tensor.shape)}")
                _first_dispatch_logged = True
            return result
        except RuntimeError as exc:
            _runtime_disabled = True
            if not _runtime_failure_warned:
                print(f"Fused Metal GroupNorm+SiLU failed; using PyTorch: {exc}")
                _runtime_failure_warned = True

    _fallback_count += 1
    return F.silu(norm(input_tensor))


def _can_dispatch_geglu(projected):
    if _geglu_runtime_disabled:
        return False
    if os.environ.get("A1111_MPS_DISABLE_FUSED_GEGLU") == "1":
        return False
    if projected.device.type != "mps" or projected.dtype != torch.float16:
        return False
    if projected.ndim != 3 or projected.shape[-1] % 2 != 0 or not projected.is_contiguous():
        return False
    if torch.is_grad_enabled() and projected.requires_grad:
        return False
    from modules import shared

    if not getattr(shared.opts, "mps_fused_geglu", True):
        return False
    return mps_flash_attention.is_available()


def geglu(input_tensor, projection):
    """Run the model's projection normally, then fuse GEGLU's GELU and multiply."""
    global _geglu_dispatch_count, _geglu_fallback_count
    global _geglu_runtime_failure_warned, _geglu_first_dispatch_logged
    global _geglu_runtime_disabled, _geglu_lut

    projected = projection(input_tensor)
    if _can_dispatch_geglu(projected):
        try:
            if _geglu_lut is None or _geglu_lut.device != projected.device:
                half_values = np.arange(65536, dtype=np.uint16).view(np.float16).copy()
                half_values = torch.from_numpy(half_values).to(projected.device)
                _geglu_lut = F.gelu(half_values).contiguous()
            result = mps_flash_attention._extension.fused_geglu_forward(projected, _geglu_lut)
            _geglu_dispatch_count += 1
            if not _geglu_first_dispatch_logged:
                print(f"Fused Metal GEGLU first dispatch: {tuple(projected.shape)}")
                _geglu_first_dispatch_logged = True
            return result
        except RuntimeError as exc:
            _geglu_runtime_disabled = True
            if not _geglu_runtime_failure_warned:
                print(f"Fused Metal GEGLU failed; using PyTorch: {exc}")
                _geglu_runtime_failure_warned = True

    _geglu_fallback_count += 1
    value, gate = projected.chunk(2, dim=-1)
    return value * F.gelu(gate)


def diagnostics():
    return {
        "dispatches": _dispatch_count,
        "fallbacks": _fallback_count,
        "geglu_dispatches": _geglu_dispatch_count,
        "geglu_fallbacks": _geglu_fallback_count,
    }
