"""Opt-in coarse stage profiling for Apple Silicon inference.

Set A1111_MPS_PROFILE=1 before launch to enable. The normal path never calls
torch.mps.synchronize(); profiling synchronizes only at request/stage boundaries.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import platform
import time

import psutil
import torch


ENVIRONMENT_VARIABLE = "A1111_MPS_PROFILE"
OUTPUT_PREFIX = "MPS_PROFILE_JSON "

_active_session: ContextVar[ProfileSession | None] = ContextVar("mps_profile_session", default=None)


def requested():
    return os.environ.get(ENVIRONMENT_VARIABLE) == "1"


def active():
    return _active_session.get() is not None


def _mps_available():
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and torch.backends.mps.is_available()
    )


def _synchronize():
    torch.mps.synchronize()


def _memory_snapshot():
    process_rss = psutil.Process().memory_info().rss
    system_memory = psutil.virtual_memory()
    snapshot = {
        "process_rss": process_rss,
        "system_available": system_memory.available,
    }

    if _mps_available():
        for name in ("current_allocated_memory", "driver_allocated_memory"):
            function = getattr(torch.mps, name, None)
            if function is not None:
                try:
                    snapshot[f"mps_{name}"] = function()
                except Exception:
                    pass

    return snapshot


class ProfileSession:
    def __init__(self, metadata):
        self.metadata = metadata
        self.has_mps = _mps_available()
        self.started = None
        self.stages = defaultdict(lambda: {"calls": 0, "wall_ms": 0.0})
        self.unet_calls = 0
        self.unet_batches = 0
        self.unet_shapes = Counter()
        self.memory_start = None
        self.memory_end = None
        self.memory_peaks = {}
        self.synchronization_error = None

    def synchronize(self):
        if not self.has_mps or self.synchronization_error is not None:
            return

        try:
            _synchronize()
        except Exception as exc:
            self.synchronization_error = str(exc)

    def observe_memory(self, snapshot=None):
        snapshot = snapshot or _memory_snapshot()
        for name, value in snapshot.items():
            self.memory_peaks[name] = max(self.memory_peaks.get(name, 0), value)
        return snapshot

    def start(self):
        self.synchronize()
        self.memory_start = self.observe_memory()
        self.started = time.perf_counter()

    def record_stage(self, name, elapsed_ms):
        record = self.stages[name]
        record["calls"] += 1
        record["wall_ms"] += elapsed_ms
        self.observe_memory()

    def record_unet(self, input_tensor):
        self.unet_calls += 1
        shape = tuple(input_tensor.shape)
        if shape:
            self.unet_batches += shape[0]
        self.unet_shapes["x".join(map(str, shape))] += 1

    def finish(self, error=None):
        self.synchronize()
        total_ms = (time.perf_counter() - self.started) * 1000
        self.memory_end = self.observe_memory()
        stages = {
            name: {
                "calls": record["calls"],
                "wall_ms": round(record["wall_ms"], 3),
            }
            for name, record in sorted(self.stages.items())
        }
        accounted_ms = sum(record["wall_ms"] for record in self.stages.values())
        report = {
            "metadata": self.metadata,
            "total_wall_ms": round(total_ms, 3),
            "accounted_stage_ms": round(accounted_ms, 3),
            "unaccounted_wall_ms": round(max(total_ms - accounted_ms, 0.0), 3),
            "stages": stages,
            "unet": {
                "calls": self.unet_calls,
                "total_batch_elements": self.unet_batches,
                "input_shapes": dict(sorted(self.unet_shapes.items())),
            },
            "memory_bytes": {
                "start": self.memory_start,
                "end": self.memory_end,
                "sampled_peaks": self.memory_peaks,
            },
        }
        if error is not None:
            report["error"] = type(error).__name__
        if self.synchronization_error is not None:
            report["synchronization_error"] = self.synchronization_error

        stage_summary = ", ".join(
            f"{name}={record['wall_ms']:.1f}ms/{record['calls']}"
            for name, record in sorted(self.stages.items())
        )
        print(
            "MPS stage profile: "
            f"total={total_ms:.1f}ms, {stage_summary or 'no stages'}, "
            f"unet_calls={self.unet_calls}, unaccounted={max(total_ms - accounted_ms, 0.0):.1f}ms"
        )
        print(OUTPUT_PREFIX + json.dumps(report, sort_keys=True, separators=(",", ":"), default=str))


def metadata_for_processing(processing):
    model = getattr(processing, "sd_model", None)
    return {
        "batch_size": getattr(processing, "batch_size", None),
        "cfg_scale": getattr(processing, "cfg_scale", None),
        "height": getattr(processing, "height", None),
        "model_hash": getattr(processing, "sd_model_hash", None) or getattr(model, "sd_model_hash", None),
        "n_iter": getattr(processing, "n_iter", None),
        "sampler": getattr(processing, "sampler_name", None),
        "scheduler": getattr(processing, "scheduler", None),
        "steps": getattr(processing, "steps", None),
        "width": getattr(processing, "width", None),
    }


@contextmanager
def request(processing):
    if not requested() or active():
        yield
        return

    session = ProfileSession(metadata_for_processing(processing))
    token = _active_session.set(session)
    try:
        session.start()
    except Exception as exc:
        _active_session.reset(token)
        print(f"MPS stage profiling unavailable; continuing without it: {exc}")
        yield
        return

    error = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            try:
                session.finish(error)
            except Exception as exc:
                print(f"MPS stage profiling report failed; generation result is unchanged: {exc}")
        finally:
            _active_session.reset(token)


@contextmanager
def stage(name):
    session = _active_session.get()
    if session is None:
        yield
        return

    session.synchronize()
    started = time.perf_counter()
    try:
        yield
    finally:
        session.synchronize()
        session.record_stage(name, (time.perf_counter() - started) * 1000)


def unet_call(function, input_tensor, *args, **kwargs):
    session = _active_session.get()
    if session is not None:
        session.record_unet(input_tensor)
    return function(input_tensor, *args, **kwargs)
