#!/usr/bin/env python3
"""Benchmark an isolated FP16 SD1 ResBlock runner against the existing fused path."""

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


SD1_SHAPES = (
    (4096, 320),
    (1024, 640),
    (256, 1280),
)
EMBEDDING_CHANNELS = 1280


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    return parser.parse_args()


def measure(operation, warmup, repeats):
    with torch.inference_mode():
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


class ReferenceResBlock(torch.nn.Module):
    """Module-style equivalent of the existing fused SD1 inference path."""

    def __init__(self, channels):
        super().__init__()
        self.in_norm = torch.nn.GroupNorm(32, channels)
        self.in_conv = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.emb_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(EMBEDDING_CHANNELS, channels),
        )
        self.out_norm = torch.nn.GroupNorm(32, channels)
        self.out_conv = torch.nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, source, embedding):
        hidden = self.in_conv(mps_fused_ops.group_norm_silu(source, self.in_norm))
        embedding = self.emb_layers(embedding).to(dtype=hidden.dtype)
        hidden = mps_fused_ops.group_norm_silu_add_embedding(hidden, embedding, self.out_norm)
        return source + self.out_conv(hidden)


class ParameterBoundResBlockRunner:
    """First runner boundary: bind tensors once, retain the same fused operators."""

    def __init__(self, reference):
        self.in_norm = reference.in_norm
        self.in_conv_weight = reference.in_conv.weight
        self.in_conv_bias = reference.in_conv.bias
        self.emb_activation = reference.emb_layers[0]
        self.emb_weight = reference.emb_layers[1].weight
        self.emb_bias = reference.emb_layers[1].bias
        self.out_norm = reference.out_norm
        self.out_conv_weight = reference.out_conv.weight
        self.out_conv_bias = reference.out_conv.bias

    def __call__(self, source, embedding):
        hidden = mps_fused_ops.group_norm_silu(source, self.in_norm)
        hidden = F.conv2d(hidden, self.in_conv_weight, self.in_conv_bias, padding=1)
        embedding = F.silu(embedding)
        embedding = F.linear(embedding, self.emb_weight, self.emb_bias).to(dtype=hidden.dtype)
        hidden = mps_fused_ops.group_norm_silu_add_embedding(hidden, embedding, self.out_norm)
        hidden = F.conv2d(hidden, self.out_conv_weight, self.out_conv_bias, padding=1)
        return source + hidden


def benchmark_shape(batch, tokens, channels, warmup, repeats):
    side = int(tokens**0.5)
    reference = ReferenceResBlock(channels).eval().half().to("mps")
    runner = ParameterBoundResBlockRunner(reference)
    source = torch.randn((batch, channels, side, side), device="mps", dtype=torch.float16)
    embedding = torch.randn((batch, EMBEDDING_CHANNELS), device="mps", dtype=torch.float16)

    with torch.inference_mode():
        expected = reference(source, embedding) + 0
        actual = runner(source, embedding) + 0
        torch.mps.synchronize()

    difference = (actual.float() - expected.float()).abs()
    return {
        "reference_ms": measure(lambda: reference(source, embedding), warmup, repeats),
        "runner_ms": measure(lambda: runner(source, embedding), warmup, repeats),
        "max_error": difference.max().item(),
        "mean_error": difference.mean().item(),
        "exact": torch.equal(actual, expected),
    }


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch installation.")

    torch.manual_seed(1)
    print(f"PyTorch {torch.__version__}; FP16 MPS ResBlock runner; batch={args.batch}")
    print("Benchmark only: both paths use the existing fused GroupNorm/SiLU kernels.")
    for tokens, channels in SD1_SHAPES:
        result = benchmark_shape(args.batch, tokens, channels, args.warmup, args.repeats)
        speedup = result["reference_ms"] / result["runner_ms"]
        print(
            f"tokens={tokens:4d} channels={channels:4d} "
            f"reference={result['reference_ms']:.3f}ms runner={result['runner_ms']:.3f}ms "
            f"speedup={speedup:.3f}x max_error={result['max_error']:.6f} "
            f"mean_error={result['mean_error']:.6f} exact={result['exact']}"
        )
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
