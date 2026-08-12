#!/usr/bin/env python3
"""Benchmark merged FP16 attention projections without changing model execution."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch
import torch.nn.functional as F


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

SD1_SHAPES = (
    (4096, 320),
    (1024, 640),
    (256, 1280),
)
CONTEXT_TOKENS = 77
CONTEXT_DIM = 768


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    return parser.parse_args()


def measure(operation, warmup, repeats):
    for _ in range(warmup):
        operation()
    torch.mps.synchronize()

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        torch.mps.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


def max_difference(separate, merged):
    return max(
        (left.float() - right.float()).abs().max().item()
        for left, right in zip(separate, merged)
    )


def benchmark_self_attention(batch, tokens, channels, warmup, repeats):
    sequence = torch.randn((batch, tokens, channels), device="mps", dtype=torch.float16)
    weights = [torch.randn((channels, channels), device="mps", dtype=torch.float16) for _ in range(3)]
    merged_weight = torch.cat(weights, dim=0).contiguous()

    def separate():
        return tuple(F.linear(sequence, weight) for weight in weights)

    def merged():
        return tuple(F.linear(sequence, merged_weight).chunk(3, dim=-1))

    separate_output = separate()
    merged_output = merged()
    torch.mps.synchronize()
    return {
        "separate_ms": measure(separate, warmup, repeats),
        "merged_ms": measure(merged, warmup, repeats),
        "max_error": max_difference(separate_output, merged_output),
        "exact": all(torch.equal(left, right) for left, right in zip(separate_output, merged_output)),
    }


def benchmark_cross_attention(batch, tokens, channels, warmup, repeats):
    context = torch.randn((batch, CONTEXT_TOKENS, CONTEXT_DIM), device="mps", dtype=torch.float16)
    weights = [torch.randn((channels, CONTEXT_DIM), device="mps", dtype=torch.float16) for _ in range(2)]
    merged_weight = torch.cat(weights, dim=0).contiguous()

    def separate():
        return tuple(F.linear(context, weight) for weight in weights)

    def merged():
        return tuple(F.linear(context, merged_weight).chunk(2, dim=-1))

    separate_output = separate()
    merged_output = merged()
    torch.mps.synchronize()
    return {
        "separate_ms": measure(separate, warmup, repeats),
        "merged_ms": measure(merged, warmup, repeats),
        "max_error": max_difference(separate_output, merged_output),
        "exact": all(torch.equal(left, right) for left, right in zip(separate_output, merged_output)),
    }


def report(kind, tokens, channels, result):
    speedup = result["separate_ms"] / result["merged_ms"]
    print(
        f"{kind:5s} tokens={tokens:4d} channels={channels:4d} "
        f"separate={result['separate_ms']:.3f}ms merged={result['merged_ms']:.3f}ms "
        f"speedup={speedup:.3f}x max_error={result['max_error']:.6f} exact={result['exact']}"
    )


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch installation.")

    torch.manual_seed(1)
    print(f"PyTorch {torch.__version__}; FP16 MPS projection fusion; batch={args.batch}")
    print("Benchmark only: merged weights are prepared before timing and no model path is modified.")
    for tokens, channels in SD1_SHAPES:
        self_result = benchmark_self_attention(args.batch, tokens, channels, args.warmup, args.repeats)
        cross_result = benchmark_cross_attention(args.batch, tokens, channels, args.warmup, args.repeats)
        report("self", tokens, channels, self_result)
        report("cross", tokens, channels, cross_result)
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
