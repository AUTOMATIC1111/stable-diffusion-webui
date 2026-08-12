#!/usr/bin/env python3
"""Benchmark direct int8-weight Metal projections against MPS FP16 linear."""

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
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    return parser.parse_args()


def quantize_per_output_channel(weight):
    scale = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    quantized = torch.round(weight / scale[:, None]).clamp(-128, 127).to(torch.int8)
    return quantized.contiguous(), scale.to(dtype=torch.float16).contiguous()


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


def benchmark_shape(rows, features, warmup, repeats, extension):
    torch.manual_seed(rows + features)
    activation = torch.randn((rows, features), device="mps", dtype=torch.float16)
    weight = torch.randn((features, features), device="mps", dtype=torch.float16)
    quantized_weight, scale = quantize_per_output_channel(weight)

    expected = F.linear(activation, weight)
    quantized_expected = F.linear(activation, quantized_weight.half() * scale[:, None])
    scalar_actual = extension.quantized_linear_forward(activation, quantized_weight, scale)
    simd_actual = extension.quantized_linear_simd_forward(activation, quantized_weight, scale)
    torch.mps.synchronize()
    scalar_kernel_difference = (scalar_actual.float() - quantized_expected.float()).abs()
    simd_kernel_difference = (simd_actual.float() - quantized_expected.float()).abs()
    quantization_difference = (simd_actual.float() - expected.float()).abs()

    fp16_ms = measure(lambda: F.linear(activation, weight), warmup, repeats)
    scalar_int8_ms = measure(
        lambda: extension.quantized_linear_forward(activation, quantized_weight, scale),
        warmup,
        repeats,
    )
    simd_int8_ms = measure(
        lambda: extension.quantized_linear_simd_forward(activation, quantized_weight, scale),
        warmup,
        repeats,
    )
    return {
        "fp16_ms": fp16_ms,
        "scalar_int8_ms": scalar_int8_ms,
        "simd_int8_ms": simd_int8_ms,
        "scalar_kernel_max_error": scalar_kernel_difference.max().item(),
        "simd_kernel_max_error": simd_kernel_difference.max().item(),
        "quantization_max_error": quantization_difference.max().item(),
        "quantization_mean_error": quantization_difference.mean().item(),
        "fp16_weight_bytes": weight.numel() * weight.element_size(),
        "int8_weight_bytes": quantized_weight.numel() * quantized_weight.element_size() + scale.numel() * scale.element_size(),
    }


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch installation.")

    try:
        import metal_flash_sdpa
    except ImportError as exc:
        raise SystemExit("Install the optional Metal extension before running this benchmark.") from exc

    if not getattr(metal_flash_sdpa, "A1111_MPS_QUANTIZED_LINEAR", False):
        raise SystemExit("The installed Metal extension does not contain the quantized linear proof kernel.")

    extension = metal_flash_sdpa
    print(f"PyTorch {torch.__version__}; int8 weights with per-output-channel fp16 scales; MPS")
    print("Acceptance gate: low error and simd_int8_ms approaches fp16_ms on representative shapes.")
    for rows, features in SD1_SHAPES:
        result = benchmark_shape(rows, features, args.warmup, args.repeats, extension)
        compression = result["fp16_weight_bytes"] / result["int8_weight_bytes"]
        print(
            f"rows={rows:4d} features={features:4d} "
            f"fp16={result['fp16_ms']:.3f}ms "
            f"scalar_int8={result['scalar_int8_ms']:.3f}ms "
            f"simd_int8={result['simd_int8_ms']:.3f}ms "
            f"simd_slowdown={result['simd_int8_ms'] / result['fp16_ms']:.2f}x "
            f"simd_kernel_max_error={result['simd_kernel_max_error']:.4f} "
            f"quantization_max_error={result['quantization_max_error']:.4f} "
            f"quantization_mean_error={result['quantization_mean_error']:.4f} "
            f"weight_compression={compression:.2f}x"
        )


if __name__ == "__main__":
    main()
