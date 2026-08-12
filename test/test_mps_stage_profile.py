from contextlib import redirect_stdout
import io
import json
import os
from types import SimpleNamespace
from unittest import mock

import torch

from modules import mps_stage_profile


def processing_stub():
    return SimpleNamespace(
        batch_size=1,
        cfg_scale=1.15,
        height=640,
        n_iter=1,
        sampler_name="DPM++ SDE",
        scheduler="Karras",
        sd_model_hash="8ecad70a19",
        steps=5,
        width=384,
    )


def report_from_output(output):
    line = next(line for line in output.splitlines() if line.startswith(mps_stage_profile.OUTPUT_PREFIX))
    return json.loads(line.removeprefix(mps_stage_profile.OUTPUT_PREFIX))


def test_disabled_profile_does_not_synchronize_or_print():
    output = io.StringIO()
    source = torch.zeros((2, 4, 8, 8))
    environment = {key: value for key, value in os.environ.items() if key != mps_stage_profile.ENVIRONMENT_VARIABLE}

    with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(mps_stage_profile, "_synchronize") as synchronize:
        with redirect_stdout(output), mps_stage_profile.request(processing_stub()):
            with mps_stage_profile.stage("sampler"):
                result = mps_stage_profile.unet_call(lambda value: value + 1, source)

    assert torch.equal(result, source + 1)
    assert output.getvalue() == ""
    synchronize.assert_not_called()


def test_enabled_profile_reports_stages_unet_shapes_and_memory():
    output = io.StringIO()
    source = torch.zeros((2, 4, 80, 48))
    memory = {
        "process_rss": 100,
        "system_available": 200,
        "mps_current_allocated_memory": 300,
    }

    with mock.patch.dict(os.environ, {mps_stage_profile.ENVIRONMENT_VARIABLE: "1"}), \
            mock.patch.object(mps_stage_profile, "_mps_available", return_value=False), \
            mock.patch.object(mps_stage_profile, "_memory_snapshot", return_value=memory), \
            redirect_stdout(output):
        with mps_stage_profile.request(processing_stub()):
            with mps_stage_profile.stage("sampler"):
                result = mps_stage_profile.unet_call(lambda value: value + 1, source)

    report = report_from_output(output.getvalue())
    assert torch.equal(result, source + 1)
    assert report["metadata"]["model_hash"] == "8ecad70a19"
    assert report["stages"]["sampler"]["calls"] == 1
    assert report["unet"] == {
        "calls": 1,
        "input_shapes": {"2x4x80x48": 1},
        "total_batch_elements": 2,
    }
    assert report["memory_bytes"]["sampled_peaks"] == memory


def test_enabled_mps_profile_synchronizes_only_at_coarse_boundaries():
    output = io.StringIO()
    memory = {"process_rss": 100, "system_available": 200}

    with mock.patch.dict(os.environ, {mps_stage_profile.ENVIRONMENT_VARIABLE: "1"}), \
            mock.patch.object(mps_stage_profile, "_mps_available", return_value=True), \
            mock.patch.object(mps_stage_profile, "_memory_snapshot", return_value=memory), \
            mock.patch.object(mps_stage_profile, "_synchronize") as synchronize, \
            redirect_stdout(output):
        with mps_stage_profile.request(processing_stub()):
            with mps_stage_profile.stage("sampler"):
                pass

    assert synchronize.call_count == 4
    assert report_from_output(output.getvalue())["stages"]["sampler"]["calls"] == 1
