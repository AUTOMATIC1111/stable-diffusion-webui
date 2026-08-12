#!/usr/bin/env python3
"""Benchmark the main SD 1.x UNet operation shapes on Apple MPS."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F


SD1_SHAPES = (
    (4096, 320),
    (1024, 640),
    (256, 1280),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2, help="UNet batch size (2 includes CFG)")
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


def measure_shape(batch, tokens, channels, warmup, repeats):
    side = int(tokens**0.5)
    image = torch.randn((batch, channels, side, side), device="mps", dtype=torch.float16)
    convolution_weight = torch.randn((channels, channels, 3, 3), device="mps", dtype=torch.float16)
    convolution_bias = torch.randn((channels,), device="mps", dtype=torch.float16)
    sequence = image.flatten(2).transpose(1, 2)
    projection_weight = torch.randn((channels, channels), device="mps", dtype=torch.float16)

    heads = 8
    query = sequence.view(batch, tokens, heads, channels // heads).transpose(1, 2)

    return {
        "conv3x3": measure(
            lambda: F.conv2d(image, convolution_weight, convolution_bias, padding=1),
            warmup,
            repeats,
        ),
        "groupnorm+silu": measure(
            lambda: F.silu(F.group_norm(image, 32)),
            warmup,
            repeats,
        ),
        "linear": measure(
            lambda: F.linear(sequence, projection_weight),
            warmup,
            repeats,
        ),
        "sdpa": measure(
            lambda: F.scaled_dot_product_attention(query, query, query, dropout_p=0.0),
            warmup,
            repeats,
        ),
    }


def benchmark_shape(batch, tokens, channels, warmup, repeats):
    results = measure_shape(batch, tokens, channels, warmup, repeats)
    torch.mps.empty_cache()
    return results


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch installation.")

    torch.manual_seed(1)
    print(f"PyTorch {torch.__version__}; batch={args.batch}; float16; MPS")
    for tokens, channels in SD1_SHAPES:
        results = benchmark_shape(args.batch, tokens, channels, args.warmup, args.repeats)
        measurements = " ".join(f"{name}={milliseconds:.3f}ms" for name, milliseconds in results.items())
        print(f"tokens={tokens:4d} channels={channels:4d} {measurements}")


if __name__ == "__main__":
    main()
