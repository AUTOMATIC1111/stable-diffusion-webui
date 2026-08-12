#!/usr/bin/env python3
"""Benchmark the SD1 VAE single-head attention path without changing routing."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import torch
import torch.nn.functional as F

_benchmark_parser = argparse.ArgumentParser(add_help=False)
_benchmark_parser.add_argument("--warmup", type=int, default=2)
_benchmark_parser.add_argument("--repeats", type=int, default=5)
_benchmark_parser.add_argument("--decode-repeats", type=int, default=2)
_benchmark_args, _webui_args = _benchmark_parser.parse_known_args()
sys.argv = [sys.argv[0], *_webui_args]

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "repositories" / "k-diffusion"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "repositories" / "stable-diffusion-stability-ai"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "repositories" / "generative-models"))

import webui  # noqa: E402, F401

from modules import devices, initialize, mps_flash_attention, sd_models, sd_samplers_common, shared  # noqa: E402


def parse_args():
    return _benchmark_args


def synchronize():
    torch.mps.synchronize()


def measure(operation, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        synchronize()
        values = []
        for _ in range(repeats):
            started = time.perf_counter()
            operation()
            synchronize()
            values.append((time.perf_counter() - started) * 1000)
    return {
        "median": statistics.median(values),
        "p25": sorted(values)[max(0, len(values) // 4)],
        "p75": sorted(values)[min(len(values) - 1, (len(values) * 3) // 4)],
    }


def attention_path(norm, q_conv, k_conv, v_conv, output_conv, source, mode):
    normalized = norm(source)
    q = q_conv(normalized)
    k = k_conv(normalized)
    v = v_conv(normalized)
    batch, channels, height, width = q.shape
    tokens = height * width
    q = q.reshape(batch, channels, tokens).transpose(1, 2).unsqueeze(1)
    k = k.reshape(batch, channels, tokens).transpose(1, 2).unsqueeze(1)
    v = v.reshape(batch, channels, tokens).transpose(1, 2).unsqueeze(1)
    if mode == "fp16":
        q, k, v = q.half(), k.half(), v.half()
        attended = F.scaled_dot_product_attention(q, k, v).float()
    elif mode == "mfa":
        attended = mps_flash_attention._extension.MetalFlashAttentionForward.apply(q.half(), k.half(), v.half(), channels**-0.5, False).float()
    else:
        attended = F.scaled_dot_product_attention(q, k, v)
    attended = attended.squeeze(1).transpose(1, 2).reshape(batch, channels, height, width)
    return source + output_conv(attended.to(dtype=source.dtype))


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch installation.")

    initialize.initialize()
    if shared.sd_model is None:
        sd_models.reload_model_weights()
    model = shared.sd_model
    decoder = model.first_stage_model
    attention = decoder.decoder.mid.attn_1
    source = torch.randn((1, attention.in_channels, 64, 64), device=devices.device, dtype=devices.dtype_vae)
    latent = torch.randn((1, 4, 64, 64), device=devices.device, dtype=devices.dtype_vae)
    print(f"PyTorch {torch.__version__}; VAE attention channels={attention.in_channels}; Q/K/V=[1,1,4096,{attention.in_channels}]")

    paths = {
        "fp32": lambda: attention_path(attention.norm, attention.q, attention.k, attention.v, attention.proj_out, source, "fp32"),
        "fp16": lambda: attention_path(attention.norm, attention.q, attention.k, attention.v, attention.proj_out, source, "fp16"),
        "mfa": lambda: attention_path(attention.norm, attention.q, attention.k, attention.v, attention.proj_out, source, "mfa"),
    }
    outputs = {}
    for name, operation in paths.items():
        try:
            result = measure(operation, args.warmup, args.repeats)
            print(f"{name} attention={result}")
            with torch.inference_mode():
                outputs[name] = operation().float()
            synchronize()
        except Exception as exc:
            print(f"{name} unavailable={type(exc).__name__}: {exc}")

    if "fp32" in outputs and "fp16" in outputs:
        difference = (outputs["fp32"] - outputs["fp16"]).abs()
        print(f"fp16_attention_error max={difference.max().item():.6f} mean={difference.mean().item():.6f} exact={torch.equal(outputs['fp32'], outputs['fp16'])}")
    if "fp32" in outputs and "mfa" in outputs:
        difference = (outputs["fp32"] - outputs["mfa"]).abs()
        print(f"mfa_attention_error max={difference.max().item():.6f} mean={difference.mean().item():.6f} exact={torch.equal(outputs['fp32'], outputs['mfa'])}")

    original_forward = attention.forward
    decode_outputs = {}
    for name in ("fp32", "fp16", "mfa"):
        attention.forward = lambda value, mode=name: attention_path(
            attention.norm, attention.q, attention.k, attention.v, attention.proj_out, value, mode
        )
        try:
            with torch.inference_mode():
                decoded = sd_samplers_common.decode_first_stage(model, latent)[0].float().cpu()
            decode_outputs[name] = decoded
            print(f"full_decode_{name}=ok nan={not torch.isfinite(decoded).all().item()}")
        except Exception as exc:
            print(f"full_decode_{name}=unavailable {type(exc).__name__}: {exc}")
    attention.forward = original_forward
    if "fp32" in decode_outputs:
        for name in ("fp16", "mfa"):
            if name in decode_outputs:
                difference = (decode_outputs[name] - decode_outputs["fp32"]).abs()
                print(f"full_decode_{name}_error max={difference.max().item():.6f} mean={difference.mean().item():.6f} exact={torch.equal(decode_outputs[name], decode_outputs['fp32'])}")
    print(f"decoder_parameters={sum(parameter.numel() for parameter in decoder.parameters())}")


if __name__ == "__main__":
    main()
