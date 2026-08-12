#!/usr/bin/env python3
"""Compare native MPS SDPA with memory-bounded sliced attention."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096, help="Query/key tokens (4096 = SD 512px latent)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=40)
    parser.add_argument("--chunk", type=int, default=1024, help="Query tokens per sliced-attention chunk")
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def synchronize():
    torch.mps.synchronize()


def sliced_attention(query, key, value, chunk_size):
    scale = query.shape[-1] ** -0.5
    output = torch.empty_like(query)
    key_transposed = key.transpose(-1, -2)
    for start in range(0, query.shape[-2], chunk_size):
        end = min(start + chunk_size, query.shape[-2])
        scores = torch.matmul(query[:, :, start:end], key_transposed) * scale
        probabilities = scores.softmax(dim=-1)
        output[:, :, start:end] = torch.matmul(probabilities, value)
    return output


def measure(operation, query, key, value, repeats):
    for _ in range(2):
        result = operation(query, key, value)
        synchronize()
        del result

    timings = []
    result = None
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        result = operation(query, key, value)
        synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings), result


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch installation.")

    shape = (args.batch, args.heads, args.tokens, args.head_dim)
    torch.manual_seed(1)
    query = torch.randn(shape, device="mps", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    native_ms, native_result = measure(
        lambda q, k, v: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0),
        query,
        key,
        value,
        args.repeats,
    )
    sliced_ms, sliced_result = measure(
        lambda q, k, v: sliced_attention(q, k, v, args.chunk),
        query,
        key,
        value,
        args.repeats,
    )

    difference = (native_result.float() - sliced_result.float()).abs()
    fastest = "native SDPA" if native_ms <= sliced_ms else "sliced"
    speedup = max(native_ms, sliced_ms) / min(native_ms, sliced_ms)

    print(f"PyTorch: {torch.__version__}")
    print(f"Shape: {shape}; dtype: float16")
    print(f"Native MPS SDPA: {native_ms:.2f} ms")
    print(f"Sliced attention: {sliced_ms:.2f} ms")
    print(f"Fastest: {fastest} ({speedup:.2f}x)")
    print(f"Difference: max={difference.max().item():.6f}, mean={difference.mean().item():.6f}")


if __name__ == "__main__":
    main()
