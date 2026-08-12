#!/usr/bin/env python3
"""Measure a standalone fused Metal GEGLU at exact SD 1.x UNet shapes."""

from __future__ import annotations

import argparse
import importlib
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F


SHAPES = ((4096, 320, 5), (1024, 640, 5), (256, 1280, 6))


def compile_extension():
    module = importlib.import_module("metal_flash_sdpa")
    if not getattr(module, "A1111_MPS_FUSED_GEGLU", False):
        raise RuntimeError("installed Metal extension does not include fused GEGLU")
    return module


def reference(projected):
    value, gate = projected.chunk(2, dim=-1)
    return value * F.gelu(gate)


def timed(operation):
    torch.mps.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.mps.synchronize()
    return (time.perf_counter() - started) * 1000, result


def timed_series(operation, calls):
    torch.mps.synchronize()
    started = time.perf_counter()
    result = None
    for _ in range(calls):
        result = operation()
    torch.mps.synchronize()
    return (time.perf_counter() - started) * 1000, result


def measure(extension, gelu_lut, batch, tokens, channels, calls, repeats):
    inner = channels * 4
    generator = torch.Generator(device="cpu").manual_seed(600100635 + batch + channels)
    # A real linear projection has roughly unit-scale activations after trained weights.
    projected = torch.randn((batch, tokens, inner * 2), generator=generator)
    projected = projected.to(device="mps", dtype=torch.float16).contiguous()

    first_reference, expected = timed(lambda: reference(projected))
    first_fused, actual = timed(lambda: extension.fused_geglu_forward(projected, gelu_lut))
    expected_float = expected.float().cpu()
    actual_float = actual.float().cpu()
    difference = (expected_float - actual_float).abs()
    parity = {
        "mean_abs": difference.mean().item(),
        "max_abs": difference.max().item(),
        "cosine": F.cosine_similarity(expected_float.flatten(), actual_float.flatten(), dim=0).item(),
        "finite": bool(torch.isfinite(actual_float).all()),
    }
    parity["passed"] = (
        parity["finite"]
        and parity["mean_abs"] <= 0.001
        and parity["max_abs"] <= 0.02
        and parity["cosine"] >= 0.9999
    )

    for _ in range(5):
        reference(projected)
        extension.fused_geglu_forward(projected, gelu_lut)
    torch.mps.synchronize()

    reference_times = []
    fused_times = []
    for index in range(repeats):
        if index % 2:
            fused_times.append(timed(lambda: extension.fused_geglu_forward(projected, gelu_lut))[0])
            reference_times.append(timed(lambda: reference(projected))[0])
        else:
            reference_times.append(timed(lambda: reference(projected))[0])
            fused_times.append(timed(lambda: extension.fused_geglu_forward(projected, gelu_lut))[0])

    reference_median = statistics.median(reference_times)
    fused_median = statistics.median(fused_times)

    series_reference = []
    series_fused = []
    series_repeats = max(8, repeats // 3)
    for index in range(series_repeats):
        if index % 2:
            series_fused.append(timed_series(lambda: extension.fused_geglu_forward(projected, gelu_lut), calls)[0])
            series_reference.append(timed_series(lambda: reference(projected), calls)[0])
        else:
            series_reference.append(timed_series(lambda: reference(projected), calls)[0])
            series_fused.append(timed_series(lambda: extension.fused_geglu_forward(projected, gelu_lut), calls)[0])
    series_reference_median = statistics.median(series_reference)
    series_fused_median = statistics.median(series_fused)
    return {
        "batch": batch,
        "tokens": tokens,
        "channels": channels,
        "first_reference": first_reference,
        "first_fused": first_fused,
        "reference_median": reference_median,
        "fused_median": fused_median,
        "saving": reference_median - fused_median,
        "speedup": (reference_median - fused_median) / reference_median * 100,
        "calls": calls,
        "series_reference": series_reference_median,
        "series_fused": series_fused_median,
        "parity": parity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    arguments = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required")

    build_started = time.perf_counter()
    extension = compile_extension()
    half_values = np.arange(65536, dtype=np.uint16).view(np.float16).copy()
    gelu_lut = F.gelu(torch.from_numpy(half_values).to("mps")).contiguous()
    print(f"Extension build/import: {time.perf_counter() - build_started:.2f} s")
    print("Reference: projected.chunk(2) followed by value * torch.nn.functional.gelu(gate)")

    results = []
    for batch in (2, 1):
        evaluations = 5 if batch == 2 else 4
        for tokens, channels, count in SHAPES:
            result = measure(extension, gelu_lut, batch, tokens, channels, count * evaluations, arguments.repeats)
            result["blocks"] = count
            results.append(result)
            parity = result["parity"]
            print(
                f"batch={batch} shape={tokens}x{channels} blocks={count} "
                f"PyTorch={result['reference_median']:.3f}ms fused={result['fused_median']:.3f}ms "
                f"speedup={result['speedup']:.1f}% "
                f"mean_abs={parity['mean_abs']:.6f} max_abs={parity['max_abs']:.6f} "
                f"cosine={parity['cosine']:.8f} {'PASS' if parity['passed'] else 'FAIL'}"
            )
            print(
                f"  coalesced {result['calls']}-call workload: "
                f"PyTorch={result['series_reference']:.2f}ms fused={result['series_fused']:.2f}ms"
            )

    pytorch_total = sum(item["series_reference"] for item in results)
    fused_total = sum(item["series_fused"] for item in results)
    saving = pytorch_total - fused_total
    speedup = saving / pytorch_total * 100
    parity_passed = all(item["parity"]["passed"] for item in results)
    # Integration is only worthwhile if the kernel itself is materially faster
    # and saves at least 50 ms in the exact nine-call workload estimate.
    gate = parity_passed and speedup >= 20 and saving >= 50
    print("\nNine-evaluation DPM++ SDE estimate")
    print(f"  PyTorch GEGLU activation total: {pytorch_total:.1f} ms")
    print(f"  Fused GEGLU activation total:   {fused_total:.1f} ms")
    print(f"  Estimated saving:               {saving:.1f} ms ({speedup:.1f}%)")
    print(f"  Numerical parity:               {'PASS' if parity_passed else 'FAIL'}")
    print(f"  RESULT: {'PASS — run an end-to-end integration A/B' if gate else 'FAIL — do not integrate'}")
    raise SystemExit(0 if gate else 2)


if __name__ == "__main__":
    main()
