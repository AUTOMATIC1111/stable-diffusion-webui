"""Opt-in capture of one real SD1 UNet call for native-backend experiments."""

import json
import os
from pathlib import Path
import statistics
import threading
import time

import torch
from safetensors.torch import save_file


ENVIRONMENT_VARIABLE = "A1111_MPS_CAPTURE_UNET"
BATCH_ENVIRONMENT_VARIABLE = "A1111_MPS_CAPTURE_UNET_BATCH"
BENCHMARK_RUNS_ENVIRONMENT_VARIABLE = "A1111_MPS_CAPTURE_UNET_BENCHMARK_RUNS"
FORMAT_VERSION = 1

_capture_lock = threading.Lock()
_capture_claimed = False


def enabled():
    return bool(os.environ.get(ENVIRONMENT_VARIABLE))


def _claim(input_tensor):
    global _capture_claimed

    if not enabled():
        return False

    try:
        requested_batch = int(os.environ.get(BATCH_ENVIRONMENT_VARIABLE, "2"))
    except ValueError:
        requested_batch = 2

    if input_tensor.ndim == 0 or input_tensor.shape[0] != requested_batch:
        return False

    with _capture_lock:
        if _capture_claimed:
            return False
        _capture_claimed = True
        return True


def _tensor_key(path):
    return "condition." + ".".join(str(part).replace("%", "%25").replace(".", "%2E") for part in path)


def _flatten(value, path, tensors):
    if isinstance(value, torch.Tensor):
        key = _tensor_key(path)
        tensors[key] = value.detach().to("cpu").contiguous()
        return {"type": "tensor", "key": key}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [[str(key), _flatten(item, (*path, key), tensors)] for key, item in value.items()],
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [_flatten(item, (*path, index), tensors) for index, item in enumerate(value)],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": "literal", "value": value}
    return {"type": "unsupported", "python_type": type(value).__qualname__, "repr": repr(value)}


def _model_metadata():
    try:
        from modules import shared

        model = shared.sd_model
        checkpoint = getattr(model, "sd_checkpoint_info", None)
        return {
            "checkpoint_filename": getattr(checkpoint, "filename", None),
            "checkpoint_sha256": getattr(checkpoint, "sha256", None),
            "checkpoint_shorthash": getattr(checkpoint, "shorthash", None),
            "model_class": type(model).__qualname__ if model is not None else None,
        }
    except Exception as error:
        return {"metadata_error": f"{type(error).__name__}: {error}"}


def _validation(reference, replay):
    reference_float = reference.detach().float().to("cpu")
    replay_float = replay.detach().float().to("cpu")
    difference = (reference_float - replay_float).abs()
    return {
        "exact": bool(torch.equal(reference.detach().to("cpu"), replay.detach().to("cpu"))),
        "mean_absolute_error": float(difference.mean().item()),
        "maximum_absolute_error": float(difference.max().item()),
    }


def _synchronize(device):
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _benchmark(run_again, device):
    try:
        measured_runs = int(os.environ.get(BENCHMARK_RUNS_ENVIRONMENT_VARIABLE, "0"))
    except ValueError:
        measured_runs = 0
    if measured_runs < 1:
        return None

    run_again()
    _synchronize(device)
    milliseconds = []
    for _ in range(measured_runs):
        _synchronize(device)
        started = time.perf_counter()
        run_again()
        _synchronize(device)
        milliseconds.append((time.perf_counter() - started) * 1000)

    return {
        "runs": measured_runs,
        "median_ms": statistics.median(milliseconds),
        "minimum_ms": min(milliseconds),
        "maximum_ms": max(milliseconds),
        "all_ms": milliseconds,
    }


def capture_and_validate(run_again, input_tensor, timestep, condition, reference_output):
    """Capture the first requested batch and immediately prove it replays in PyTorch."""
    if not _claim(input_tensor):
        return

    destination = Path(os.environ[ENVIRONMENT_VARIABLE]).expanduser()
    if destination.suffix != ".safetensors":
        print(f"UNet capture skipped: {ENVIRONMENT_VARIABLE} must name a .safetensors file")
        return
    if destination.exists():
        print(f"UNet capture skipped: destination already exists: {destination}")
        return

    try:
        replay_output = run_again()
        benchmark = _benchmark(run_again, input_tensor.device)
        tensors = {
            "input.latent": input_tensor.detach().to("cpu").contiguous(),
            "input.timestep": timestep.detach().to("cpu").contiguous(),
            "output.reference": reference_output.detach().to("cpu").contiguous(),
            "output.replay": replay_output.detach().to("cpu").contiguous(),
        }
        condition_descriptor = _flatten(condition, (), tensors)
        validation = _validation(reference_output, replay_output)
        metadata = {
            "format_version": str(FORMAT_VERSION),
            "captured_unix_time": str(time.time()),
            "condition_descriptor": json.dumps(condition_descriptor, separators=(",", ":")),
            "model": json.dumps(_model_metadata(), separators=(",", ":")),
            "validation": json.dumps(validation, separators=(",", ":")),
        }
        if benchmark is not None:
            metadata["pytorch_benchmark"] = json.dumps(benchmark, separators=(",", ":"))

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, destination)
        print(
            f"UNet capture saved: {destination} "
            f"(exact_replay={validation['exact']}, max_abs={validation['maximum_absolute_error']:.8g})"
        )
        if benchmark is not None:
            print(
                f"PyTorch UNet probe: median={benchmark['median_ms']:.3f}ms "
                f"min={benchmark['minimum_ms']:.3f}ms max={benchmark['maximum_ms']:.3f}ms "
                f"runs={benchmark['runs']}"
            )
    except Exception as error:
        print(f"UNet capture failed without affecting generation: {type(error).__name__}: {error}")


def reset_for_tests():
    global _capture_claimed
    with _capture_lock:
        _capture_claimed = False
