#!/usr/bin/env python3
"""Benchmark the isolated native SD1 ResBlock second-half operation."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from modules import mps_fused_ops


SD1_SHAPES = ((4096, 320), (1024, 640), (256, 1280))


def measure(operation, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        torch.mps.synchronize()
        values = []
        for _ in range(repeats):
            started = time.perf_counter()
            operation()
            torch.mps.synchronize()
            values.append((time.perf_counter() - started) * 1000)
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch installation.")
    import metal_flash_sdpa

    if not getattr(metal_flash_sdpa, "A1111_MPS_FUSED_RESBLOCK", False):
        raise SystemExit("The installed Metal extension does not contain the isolated ResBlock operation.")

    torch.manual_seed(1)
    print(f"PyTorch {torch.__version__}; FP16 MPS ResBlock second half; batch={args.batch}")
    print("Reference: existing fused GroupNorm+SiLU+embedding, PyTorch convolution, residual add.")
    print("Native: same command buffer, Metal statistics, convolution, and residual output.")

    for tokens, channels in SD1_SHAPES:
        side = int(tokens**0.5)
        hidden = torch.randn((args.batch, channels, side, side), device="mps", dtype=torch.float16)
        embedding = torch.randn((args.batch, channels), device="mps", dtype=torch.float16)
        norm = torch.nn.GroupNorm(32, channels).eval().half().to("mps")
        conv = torch.nn.Conv2d(channels, channels, 3, padding=1).eval().half().to("mps")
        residual = torch.randn_like(hidden)

        def reference(hidden=hidden, embedding=embedding, norm=norm, residual=residual, conv=conv):
            normalized = mps_fused_ops.group_norm_silu_add_embedding(hidden, embedding, norm)
            return residual + F.conv2d(normalized, conv.weight, conv.bias, padding=1)

        def native(hidden=hidden, embedding=embedding, norm=norm, residual=residual, conv=conv):
            return metal_flash_sdpa.fused_resblock_forward(
                hidden, embedding, norm.weight, norm.bias,
                conv.weight, conv.bias, residual, 32, norm.eps,
            )

        with torch.inference_mode():
            expected = reference() + 0
            actual = native() + 0
            torch.mps.synchronize()
        difference = (actual.float() - expected.float()).abs()
        reference_ms = measure(reference, args.warmup, args.repeats)
        native_ms = measure(native, args.warmup, args.repeats)
        print(
            f"tokens={tokens:4d} channels={channels:4d} reference={reference_ms:.3f}ms "
            f"native={native_ms:.3f}ms speedup={reference_ms / native_ms:.3f}x "
            f"max_error={difference.max().item():.6f} mean_error={difference.mean().item():.6f} "
            f"exact={torch.equal(actual, expected)}"
        )
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
