#!/usr/bin/env python3
"""Inspect and validate an A1111_MPS_CAPTURE_UNET fixture."""

import argparse
import json

from safetensors import safe_open
from safetensors.torch import load_file
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", help="Path to the captured .safetensors file")
    args = parser.parse_args()

    tensors = load_file(args.capture)
    with safe_open(args.capture, framework="pt", device="cpu") as source:
        metadata = source.metadata()

    reference = tensors["output.reference"]
    replay = tensors["output.replay"]
    difference = (reference.float() - replay.float()).abs()
    result = {
        "model": json.loads(metadata["model"]),
        "latent": {"shape": list(tensors["input.latent"].shape), "dtype": str(tensors["input.latent"].dtype)},
        "timestep": {"shape": list(tensors["input.timestep"].shape), "values": tensors["input.timestep"].tolist()},
        "condition_tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in tensors.items()
            if key.startswith("condition.")
        },
        "output": {"shape": list(reference.shape), "dtype": str(reference.dtype)},
        "validation": {
            "exact": bool(torch.equal(reference, replay)),
            "mean_absolute_error": float(difference.mean().item()),
            "maximum_absolute_error": float(difference.max().item()),
        },
    }
    if "pytorch_benchmark" in metadata:
        result["pytorch_benchmark"] = json.loads(metadata["pytorch_benchmark"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
