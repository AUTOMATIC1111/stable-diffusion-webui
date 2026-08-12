import json
import os
from unittest import mock

from safetensors import safe_open
from safetensors.torch import load_file
import torch

from modules import mps_unet_capture


def test_disabled_capture_does_not_replay(tmp_path):
    destination = tmp_path / "capture.safetensors"
    environment = {key: value for key, value in os.environ.items() if key != mps_unet_capture.ENVIRONMENT_VARIABLE}
    replay = mock.Mock(return_value=torch.ones((2, 4, 8, 8)))

    with mock.patch.dict(os.environ, environment, clear=True):
        mps_unet_capture.reset_for_tests()
        mps_unet_capture.capture_and_validate(
            replay,
            torch.zeros((2, 4, 8, 8)),
            torch.ones(2),
            {"c_crossattn": [torch.zeros((2, 77, 768))]},
            torch.ones((2, 4, 8, 8)),
        )

    replay.assert_not_called()
    assert not destination.exists()


def test_capture_saves_complete_exact_replay_fixture(tmp_path):
    destination = tmp_path / "capture.safetensors"
    latent = torch.arange(2 * 4 * 8 * 8, dtype=torch.float16).reshape(2, 4, 8, 8)
    timestep = torch.tensor([1.5, 1.5], dtype=torch.float16)
    condition = {"c_crossattn": [torch.zeros((2, 77, 768), dtype=torch.float16)], "c_concat": []}
    reference = latent + 1

    with mock.patch.dict(os.environ, {mps_unet_capture.ENVIRONMENT_VARIABLE: str(destination)}, clear=False):
        mps_unet_capture.reset_for_tests()
        mps_unet_capture.capture_and_validate(lambda: latent + 1, latent, timestep, condition, reference)

    tensors = load_file(destination)
    with safe_open(destination, framework="pt", device="cpu") as source:
        metadata = source.metadata()

    assert torch.equal(tensors["input.latent"], latent)
    assert torch.equal(tensors["input.timestep"], timestep)
    assert tensors["condition.c_crossattn.0"].shape == (2, 77, 768)
    assert torch.equal(tensors["output.reference"], tensors["output.replay"])
    assert json.loads(metadata["validation"]) == {
        "exact": True,
        "mean_absolute_error": 0.0,
        "maximum_absolute_error": 0.0,
    }
    assert "pytorch_benchmark" not in metadata


def test_capture_waits_for_requested_batch(tmp_path):
    destination = tmp_path / "capture.safetensors"
    environment = {
        mps_unet_capture.ENVIRONMENT_VARIABLE: str(destination),
        mps_unet_capture.BATCH_ENVIRONMENT_VARIABLE: "2",
    }

    with mock.patch.dict(os.environ, environment, clear=False):
        mps_unet_capture.reset_for_tests()
        mps_unet_capture.capture_and_validate(
            lambda: torch.ones((1, 4, 8, 8)),
            torch.zeros((1, 4, 8, 8)),
            torch.ones(1),
            {},
            torch.ones((1, 4, 8, 8)),
        )
        assert not destination.exists()

        mps_unet_capture.capture_and_validate(
            lambda: torch.ones((2, 4, 8, 8)),
            torch.zeros((2, 4, 8, 8)),
            torch.ones(2),
            {},
            torch.ones((2, 4, 8, 8)),
        )

    assert destination.exists()
