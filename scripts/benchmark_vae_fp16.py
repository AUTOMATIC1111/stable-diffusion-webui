#!/usr/bin/env python3
"""Benchmark full FP16 SD1 VAE decode against the loaded FP32 decoder."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--warmup", type=int, default=2)
_parser.add_argument("--repeats", type=int, default=6)
_benchmark_args, _webui_args = _parser.parse_known_args()
sys.argv = [sys.argv[0], *_webui_args]

repository_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))
sys.path.insert(0, str(repository_root / "repositories" / "k-diffusion"))
sys.path.insert(0, str(repository_root / "repositories" / "stable-diffusion-stability-ai"))
sys.path.insert(0, str(repository_root / "repositories" / "generative-models"))

import webui  # noqa: E402, F401

from modules import devices, initialize, mps_fused_ops, sd_models, shared  # noqa: E402


def synchronize():
    torch.mps.synchronize()


def distribution(values):
    values = sorted(values)
    return {
        "median": round(statistics.median(values), 3),
        "p25": round(values[max(0, len(values) // 4)], 3),
        "p75": round(values[min(len(values) - 1, (len(values) * 3) // 4)], 3),
    }


def as_uint8(image):
    image = ((image.float() + 1.0) / 2.0).clamp(0.0, 1.0)
    return (image * 255.0).round().to(torch.uint8)


def parameter_bytes(module):
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def decode_with_timing(model, latent, warmup, repeats):
    def decode():
        return model.decode_first_stage(latent)

    with torch.inference_mode():
        for _ in range(warmup):
            decode()
        synchronize()
        timings = []
        output = None
        for _ in range(repeats):
            started = time.perf_counter()
            output = decode()
            synchronize()
            timings.append((time.perf_counter() - started) * 1000)
    return distribution(timings), output.detach().float().cpu()


def main():
    args = _benchmark_args
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch installation.")

    initialize.initialize()
    if shared.sd_model is None:
        sd_models.reload_model_weights()
    model = shared.sd_model
    decoder = model.first_stage_model
    fp32_state = {name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()}
    latent_fp32 = torch.randn((1, 4, 64, 64), device=devices.device, dtype=torch.float32)
    baseline_dispatches = mps_fused_ops.diagnostics().copy()

    decoder.float()
    decoder.load_state_dict(fp32_state)
    latent_fp32 = latent_fp32.to(dtype=torch.float32)
    fp32_timing, fp32_output = decode_with_timing(model, latent_fp32, args.warmup, args.repeats)
    fp32_memory = parameter_bytes(decoder)
    fp32_dispatches = mps_fused_ops.diagnostics().copy()

    decoder.half()
    latent_fp16 = latent_fp32.half()
    fp16_timing, fp16_output = decode_with_timing(model, latent_fp16, args.warmup, args.repeats)
    fp16_memory = parameter_bytes(decoder)
    fp16_dispatches = mps_fused_ops.diagnostics().copy()

    float_difference = (fp16_output - fp32_output).abs()
    uint8_difference = (as_uint8(fp16_output).to(torch.int16) - as_uint8(fp32_output).to(torch.int16)).abs()
    print(f"PyTorch {torch.__version__}; full FP16 VAE decode; latent={tuple(latent_fp32.shape)}")
    print(f"fp32={fp32_timing} fp16={fp16_timing}")
    print(f"median_gain_ms={fp32_timing['median'] - fp16_timing['median']:.3f}")
    print(f"fp32_parameter_bytes={fp32_memory} fp16_parameter_bytes={fp16_memory} reduction={(1 - fp16_memory / fp32_memory) * 100:.1f}%")
    print(f"float_error_max={float_difference.max().item():.6f} float_error_mean={float_difference.mean().item():.6f}")
    print(f"uint8_mae={uint8_difference.float().mean().item():.6f} uint8_max={uint8_difference.max().item()} changed_pixels={(uint8_difference > 0).float().mean().item() * 100:.3f}%")
    print(f"nan_fp32={not torch.isfinite(fp32_output).all().item()} nan_fp16={not torch.isfinite(fp16_output).all().item()}")
    print(f"groupnorm_dispatch_delta_fp32={fp32_dispatches['dispatches'] - baseline_dispatches['dispatches']} embedding_dispatch_delta_fp32={fp32_dispatches['embedding_dispatches'] - baseline_dispatches['embedding_dispatches']}")
    print(f"groupnorm_dispatch_delta_fp16={fp16_dispatches['dispatches'] - fp32_dispatches['dispatches']} embedding_dispatch_delta_fp16={fp16_dispatches['embedding_dispatches'] - fp32_dispatches['embedding_dispatches']}")

    decoder.float()
    decoder.load_state_dict(fp32_state)


if __name__ == "__main__":
    main()
