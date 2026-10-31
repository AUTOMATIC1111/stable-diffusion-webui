import os
import re

import pytest

from modules.sd_model_targets import (
    ALLOWED_MODEL_TARGETS,
    UnsafeModelTargetError,
    assert_model_target_allowed,
    is_model_target_allowed,
    validate_model_config_targets,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model targets from the Stable Diffusion and SDXL repositories that the
# web UI loads configs from at runtime (repositories/
# stable-diffusion-stability-ai and generative-models). These are the
# model-section targets that are not already in the web UI's own
# configs/*.yaml; they were verified against the live upstream files.
UPSTREAM_MODEL_TARGETS = [
    "ldm.models.diffusion.ddpm.ImageEmbeddingConditionedLatentDiffusion",
    "ldm.models.diffusion.ddpm.LatentDepth2ImageDiffusion",
    "ldm.models.diffusion.ddpm.LatentUpscaleDiffusion",
    "ldm.modules.diffusionmodules.upscaling.ImageConcatWithNoiseAugmentation",
    "ldm.modules.encoders.modules.CLIPEmbeddingNoiseAugmentation",
    "ldm.modules.encoders.modules.ClipImageEmbedder",
    "ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder",
    "ldm.modules.encoders.modules.FrozenOpenCLIPImageEmbedder",
    "ldm.modules.midas.api.MiDaSInference",
    "sgm.models.autoencoder.AutoencoderKL",
]


def model_section_targets(path):
    """Return the ``target`` values under the top-level ``model:`` section.

    A minimal reader for the bundled configs: only lines inside the
    top-level ``model`` section count, so targets from a config's ``data``
    section (training data modules, loggers) are never collected here.
    """

    targets = []
    section = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            top_level = re.match(r"^(\S+):\s*$", line)
            if top_level:
                section = top_level.group(1)
                continue
            if section == "model":
                match = re.search(r"target:\s*(\S+)", line)
                if match:
                    targets.append(match.group(1).strip("\"'"))
    return targets


def stock_sd1_model_config():
    """A stock SD1 model config shape, as repair_config leaves it."""

    return {
        "target": "ldm.models.diffusion.ddpm.LatentDiffusion",
        "params": {
            "use_ema": False,
            "unet_config": {
                "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "params": {"use_checkpoint": False},
            },
            "first_stage_config": {
                "target": "ldm.models.autoencoder.AutoencoderKL",
                "params": {},
            },
            "cond_stage_config": {
                "target": "ldm.modules.encoders.modules.FrozenCLIPEmbedder",
                "params": {},
            },
        },
    }


def test_stock_targets_from_bundled_configs_are_allowed():
    configs_dir = os.path.join(REPO_ROOT, "configs")
    found = []
    for name in sorted(os.listdir(configs_dir)):
        if name.endswith(".yaml"):
            found.extend(model_section_targets(os.path.join(configs_dir, name)))

    assert len(found) >= 20
    for target in found:
        assert is_model_target_allowed(target), target


def test_upstream_stock_targets_are_allowed():
    for target in UPSTREAM_MODEL_TARGETS:
        assert is_model_target_allowed(target), target


def test_stock_model_config_validates():
    validate_model_config_targets(stock_sd1_model_config())


def test_allowlist_contains_no_untrusted_families():
    assert len(ALLOWED_MODEL_TARGETS) >= 30
    for target in ALLOWED_MODEL_TARGETS:
        assert not target.startswith(("os.", "subprocess", "torch.hub", "builtins", "main.", "edit_dataset.")), target


def test_torch_hub_load_is_refused():
    with pytest.raises(UnsafeModelTargetError):
        assert_model_target_allowed("torch.hub.load")


@pytest.mark.parametrize("target", ["os.system", "os.makedirs", "os.execv", "os.remove"])
def test_os_targets_are_refused(target):
    with pytest.raises(UnsafeModelTargetError):
        assert_model_target_allowed(target)


@pytest.mark.parametrize("target", [
    "subprocess.Popen",
    "builtins.eval",
    "builtins.exec",
    "evil.pwn",
    "attacker.helper",
    "main.DataModuleFromConfig",
    "edit_dataset.EditDataset",
    "ldm.data.laion.WebDataModuleFromConfig",
    "torchvision.transforms.Resize",
    "",
    None,
    123,
])
def test_arbitrary_targets_are_refused(target):
    with pytest.raises(UnsafeModelTargetError):
        assert_model_target_allowed(target)


def test_nested_target_is_validated():
    config = stock_sd1_model_config()
    config["params"]["unet_config"]["target"] = "os.system"
    with pytest.raises(UnsafeModelTargetError):
        validate_model_config_targets(config)

    config = stock_sd1_model_config()
    config["params"]["cond_stage_config"]["target"] = "torch.hub.load"
    with pytest.raises(UnsafeModelTargetError):
        validate_model_config_targets(config)


def test_sidecar_config_falls_back_without_executing(tmp_path):
    # A downloaded pack: checkpoint plus a same-basename YAML next to it.
    # Discovery is the same-basename lookup the web UI and the API use.
    checkpoint = tmp_path / "evil.safetensors"
    checkpoint.touch()
    sidecar = tmp_path / "evil.yaml"
    sidecar.write_text("model:\n  target: torch.hub.load\n  params:\n    repo_or_dir: .\n")
    discovered = f"{os.path.splitext(str(checkpoint))[0]}.yaml"
    assert discovered == str(sidecar)
    assert os.path.exists(discovered)

    # What load_model gets from the discovered file: refused, never resolved.
    attacker_model = {"target": "torch.hub.load", "params": {"repo_or_dir": "."}}
    resolved = []

    def gated_resolve(target):
        assert_model_target_allowed(target)
        resolved.append(target)

    with pytest.raises(UnsafeModelTargetError):
        validate_model_config_targets(attacker_model)

    # load_model falls back to the config guessed from the checkpoint
    # weights instead of executing the sidecar, and that config validates.
    guessed_model = stock_sd1_model_config()
    validate_model_config_targets(guessed_model)
    gated_resolve(guessed_model["target"])
    assert resolved == ["ldm.models.diffusion.ddpm.LatentDiffusion"]


def test_api_sidecar_path_is_gated(tmp_path):
    # api.py get_sd_models reports the discovered sidecar path for a
    # checkpoint; it never loads or imports it. The load that any API
    # caller then triggers goes through load_model, which validates.
    # So a sidecar discovered through the API path must still be refused.
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    sidecar = tmp_path / "model.yaml"
    sidecar.write_text("model:\n  target: os.makedirs\n  params:\n    name: /tmp/pwned\n")

    discovered = f"{os.path.splitext(str(checkpoint))[0]}.yaml"
    assert discovered == str(sidecar)
    assert isinstance(discovered, str)

    targets = model_section_targets(discovered)
    assert targets == ["os.makedirs"]
    for target in targets:
        with pytest.raises(UnsafeModelTargetError):
            assert_model_target_allowed(target)
    assert not os.path.exists("/tmp/pwned")


def test_repair_config_injection_cannot_smuggle(tmp_path):
    # repair_config writes use_ema / use_checkpoint / attn_type into a
    # loaded config's parameters. Those repaired parameters must not let
    # a refused target through, and the target must never be constructed
    # with them.
    attacker_model = {
        "target": "torch.hub.load",
        "params": {
            "use_ema": False,
            "unet_config": {
                "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "params": {"use_checkpoint": False},
            },
        },
    }
    resolved = []

    def gated_resolve(target):
        assert_model_target_allowed(target)
        resolved.append(target)

    with pytest.raises(UnsafeModelTargetError):
        validate_model_config_targets(attacker_model)
    assert resolved == []


def test_custom_arch_opt_in():
    # The --allow-custom-model-targets escape hatch: a power user's own
    # architecture works when they explicitly opt in, and only then.
    assert not is_model_target_allowed("mymodel.MyModel")
    assert is_model_target_allowed("mymodel.MyModel", allow_custom=True)

    custom_config = {
        "target": "mymodel.MyModel",
        "params": {
            "unet_config": {"target": "mymodel.MyUNet", "params": {}},
        },
    }
    with pytest.raises(UnsafeModelTargetError):
        validate_model_config_targets(custom_config)
    validate_model_config_targets(custom_config, allow_custom=True)
