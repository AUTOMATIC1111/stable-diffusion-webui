#!/usr/bin/env python3
"""Profile postprocessing stages without changing the WebUI runtime."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

_benchmark_parser = argparse.ArgumentParser(add_help=False)
_benchmark_parser.add_argument("--width", type=int, default=512)
_benchmark_parser.add_argument("--height", type=int, default=512)
_benchmark_parser.add_argument("--tile-size", type=int, default=256)
_benchmark_parser.add_argument("--tile-overlap", type=int, default=32)
_benchmark_parser.add_argument("--warmup", type=int, default=2)
_benchmark_parser.add_argument("--repeats", type=int, default=5)
_benchmark_parser.add_argument("--realesrgan-model", default="models/RealESRGAN/RealESRGAN_x4plus_anime_6B.pth")
_benchmark_parser.add_argument("--skip-realesrgan", action="store_true")
_benchmark_parser.add_argument("--vae-channels-last", action="store_true")
_benchmark_parser.add_argument("--vae-repeats", type=int, default=6)
_benchmark_parser.add_argument("--vae-warmup", type=int, default=2)
_benchmark_args, _webui_args = _benchmark_parser.parse_known_args()
sys.argv = [sys.argv[0], *_webui_args]

repository_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root))
sys.path.insert(0, str(repository_root / "repositories" / "k-diffusion"))
sys.path.insert(0, str(repository_root / "repositories" / "stable-diffusion-stability-ai"))
sys.path.insert(0, str(repository_root / "repositories" / "generative-models"))

import webui  # noqa: E402, F401

from modules import images, modelloader, upscaler_utils  # noqa: E402


def parse_args():
    return _benchmark_args


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(operation, device, warmup, repeats):
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        synchronize(device)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            operation()
            synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings)


def make_image(width, height):
    values = np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3)
    return Image.fromarray((values % 256).astype(np.uint8), "RGB")


def report(name, milliseconds):
    print(f"{name:28s} {milliseconds:9.3f} ms")


def distribution(values):
    values = sorted(values)
    return {
        "median": statistics.median(values),
        "p25": values[max(0, len(values) // 4)],
        "p75": values[min(len(values) - 1, (len(values) * 3) // 4)],
    }


def run_vae_channels_last(args):
    from modules import devices, initialize, sd_models, sd_samplers_common, shared

    initialize.initialize()
    if shared.sd_model is None:
        sd_models.reload_model_weights()
    model = shared.sd_model
    if model is None or model.first_stage_model is None:
        raise RuntimeError("WebUI did not load a VAE model")

    latent = torch.randn((1, 4, args.height // 8, args.width // 8), device=devices.device, dtype=devices.dtype_vae)
    decoder = model.first_stage_model
    original_format = next(decoder.parameters()).data.stride()

    def decode():
        return sd_samplers_common.decode_first_stage(model, latent)[0]

    def set_layout(channels_last):
        decoder.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
        latent_layout = latent.contiguous(memory_format=torch.channels_last) if channels_last else latent.contiguous()
        return latent_layout

    results = {"contiguous": [], "channels_last": []}
    outputs = {}
    for index in range(args.vae_warmup + args.vae_repeats * 2):
        layout = "channels_last" if index % 2 else "contiguous"
        latent_layout = set_layout(layout == "channels_last")
        if index < args.vae_warmup:
            with torch.inference_mode():
                sd_samplers_common.decode_first_stage(model, latent_layout)
            continue
        started = time.perf_counter()
        with torch.inference_mode():
            output = sd_samplers_common.decode_first_stage(model, latent_layout)[0]
        synchronize(devices.device)
        elapsed = (time.perf_counter() - started) * 1000
        results[layout].append(elapsed)
        outputs.setdefault(layout, output.detach().float().cpu())

    reference = outputs["contiguous"]
    difference = (outputs["channels_last"] - reference).abs()
    parameter_layouts = {
        "4d_channels_last": 0,
        "4d_contiguous": 0,
        "4d_other": 0,
        "non4d_contiguous": 0,
        "non4d_total": 0,
    }
    for parameter in decoder.parameters():
        memory_format = parameter.data
        if memory_format.ndim == 4 and memory_format.is_contiguous(memory_format=torch.channels_last):
            parameter_layouts["4d_channels_last"] += 1
        elif memory_format.ndim == 4 and memory_format.is_contiguous():
            parameter_layouts["4d_contiguous"] += 1
        elif memory_format.ndim == 4:
            parameter_layouts["4d_other"] += 1
        else:
            parameter_layouts["non4d_contiguous"] += int(memory_format.is_contiguous())
            parameter_layouts["non4d_total"] += 1

    print("VAE channels-last A/B")
    print(f"original_first_parameter_stride={original_format}")
    print(f"parameter_layouts={parameter_layouts}")
    print(f"input_channels_last={latent.contiguous(memory_format=torch.channels_last).is_contiguous(memory_format=torch.channels_last)}")
    for layout, values in results.items():
        print(f"{layout}={distribution(values)}")
    print(f"max_pixel_difference={difference.max().item():.6f} mean_pixel_difference={difference.mean().item():.6f} exact={torch.equal(outputs['channels_last'], reference)}")
    print(f"nan_contiguous={not torch.isfinite(outputs['contiguous']).all().item()} nan_channels_last={not torch.isfinite(outputs['channels_last']).all().item()}")


def main():
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this PyTorch installation.")

    if args.vae_channels_last:
        run_vae_channels_last(args)
        return

    device = torch.device("mps")
    image = make_image(args.width, args.height)
    print(f"PyTorch {torch.__version__}; postprocessing profile; image={args.width}x{args.height}; device={device}")
    print("Warm medians; model loading and first-time compilation are excluded from timings.")

    cpu_tensor = lambda: upscaler_utils.pil_image_to_torch_bgr(image).unsqueeze(0)
    cpu_image_tensor = cpu_tensor()
    conversion_to_mps = lambda: cpu_image_tensor.to(device=device, dtype=torch.float16)
    tensor = conversion_to_mps()
    conversion_to_pil = lambda: upscaler_utils.torch_bgr_to_pil_image(tensor)
    report("PIL -> CPU tensor", measure(cpu_tensor, torch.device("cpu"), args.warmup, args.repeats))
    report("CPU tensor -> MPS", measure(conversion_to_mps, device, args.warmup, args.repeats))
    report("MPS tensor -> CPU/PIL", measure(conversion_to_pil, device, args.warmup, args.repeats))

    identity_model = torch.nn.Identity().to(device)
    gpu_tile = lambda: upscaler_utils.tiled_upscale_2(tensor, identity_model, tile_size=args.tile_size, tile_overlap=args.tile_overlap, scale=1, device=device, desc="profile")
    report("GPU tile compose", measure(gpu_tile, device, args.warmup, args.repeats))

    grid = images.split_grid(image, args.tile_size, args.tile_size, args.tile_overlap)
    pil_tiles = [tile for _, _, row in grid.tiles for _, _, tile in row]
    pil_tile_process = lambda: [tile.copy() for tile in pil_tiles]
    report("PIL tile traversal/copy", measure(pil_tile_process, torch.device("cpu"), args.warmup, args.repeats))
    report("PIL tile composition", measure(lambda: images.combine_grid(grid), torch.device("cpu"), args.warmup, args.repeats))

    if args.skip_realesrgan:
        print("RealESRGAN inference          skipped")
    else:
        model_path = pathlib.Path(args.realesrgan_model)
        if not model_path.exists():
            print(f"RealESRGAN inference          unavailable ({model_path})")
        else:
            model = modelloader.load_spandrel_model(str(model_path), device=device, prefer_half=True, expected_architecture="ESRGAN")
            model_input = tensor
            report("RealESRGAN full inference", measure(lambda: model(model_input), device, args.warmup, args.repeats))
            tiled_model = lambda: upscaler_utils.tiled_upscale_2(model_input, model, tile_size=args.tile_size, tile_overlap=args.tile_overlap, scale=4, device=device, desc="profile")
            report("RealESRGAN GPU tiled inference", measure(tiled_model, device, args.warmup, args.repeats))

    print("VAE decode                   unavailable: run this profile with a loaded WebUI VAE stage")


if __name__ == "__main__":
    main()
