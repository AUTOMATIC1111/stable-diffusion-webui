"""Allowlist for model targets in checkpoint configs.

A checkpoint config is a YAML file that names, in its ``model`` section, the
Python class to build the model from (``model.target``) plus nested targets
for the UNet, first stage model, conditioner and so on. The web UI
auto-adopts a YAML file sitting next to a checkpoint (same basename), so a
downloaded checkpoint pack can ship its own config, and every ``target`` in
it used to be imported and called with attacker-controlled parameters and
no check at all. That is arbitrary code execution on checkpoint load
(for example a config pointing at ``torch.hub.load``).

This module is the choke point. It deliberately imports nothing from the
rest of the web UI (no torch, no shared state) so it can be tested on its
own, and so importing it never executes model code.

Rules:

* A target is allowed when it is one of the classes used by the model
  configs the web UI itself ships or loads: the configs in ``configs/``,
  the Stable Diffusion configs from the ``stable-diffusion-stability-ai``
  repository (``w-e-w/stablediffusion``) and the SDXL inference configs
  from ``Stability-AI/generative-models``. The list below is the exact
  union of the ``target`` entries in the ``model`` sections of those
  configs, verified against the live upstream files.
* Targets that only appear in a config's ``data`` section (training data
  modules, image loggers, torchvision transforms) are deliberately NOT
  listed: loading a checkpoint never instantiates the ``data`` section,
  so allowing them would only widen the hole.
* Anything else is refused, including ``torch.hub.load``, ``os.*``,
  ``subprocess.*`` and any other importable callable, unless the user
  explicitly opted in with ``--allow-custom-model-targets``. That flag is
  the escape hatch for legitimate custom architectures: a power user who
  wrote their own model class and their own config passes the flag and
  their targets work unchanged. Without the flag, an untrusted sidecar
  config is refused instead of executed; ``load_model`` then falls back
  to the config guessed from the checkpoint weights.
"""

ALLOWED_MODEL_TARGETS = frozenset({
    # SD1 / SD2 / inpainting / depth / unclip / alt-diffusion, from the
    # web UI's own configs/ and from w-e-w/stablediffusion
    # configs/stable-diffusion/*.yaml (model sections only).
    "ldm.lr_scheduler.LambdaLinearScheduler",
    "ldm.models.autoencoder.AutoencoderKL",
    "ldm.models.diffusion.ddpm.ImageEmbeddingConditionedLatentDiffusion",
    "ldm.models.diffusion.ddpm.LatentDepth2ImageDiffusion",
    "ldm.models.diffusion.ddpm.LatentDiffusion",
    "ldm.models.diffusion.ddpm.LatentInpaintDiffusion",
    "ldm.models.diffusion.ddpm.LatentUpscaleDiffusion",
    "ldm.modules.diffusionmodules.openaimodel.UNetModel",
    "ldm.modules.diffusionmodules.upscaling.ImageConcatWithNoiseAugmentation",
    "ldm.modules.encoders.modules.CLIPEmbeddingNoiseAugmentation",
    "ldm.modules.encoders.modules.ClipImageEmbedder",
    "ldm.modules.encoders.modules.FrozenCLIPEmbedder",
    "ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder",
    "ldm.modules.encoders.modules.FrozenOpenCLIPImageEmbedder",
    "ldm.modules.midas.api.MiDaSInference",
    # Web UI's own model classes, from configs/*.yaml (model sections only).
    "modules.models.diffusion.ddpm_edit.LatentDiffusion",
    "modules.models.sd3.sd3_model.SD3Inferencer",
    "modules.xlmr.BertSeriesModelWithTransformation",
    "modules.xlmr_m18.BertSeriesModelWithTransformation",
    # SDXL, from the web UI's configs/*.yaml and from
    # Stability-AI/generative-models configs/inference/sd_xl_base.yaml
    # and sd_xl_refiner.yaml (model sections only).
    "sgm.models.autoencoder.AutoencoderKL",
    "sgm.models.autoencoder.AutoencoderKLInferenceWrapper",
    "sgm.models.diffusion.DiffusionEngine",
    "sgm.modules.GeneralConditioner",
    "sgm.modules.diffusionmodules.denoiser.DiscreteDenoiser",
    "sgm.modules.diffusionmodules.denoiser_scaling.EpsScaling",
    "sgm.modules.diffusionmodules.denoiser_scaling.VScaling",
    "sgm.modules.diffusionmodules.denoiser_weighting.EpsWeighting",
    "sgm.modules.diffusionmodules.denoiser_weighting.VWeighting",
    "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
    "sgm.modules.diffusionmodules.openaimodel.UNetModel",
    "sgm.modules.encoders.modules.ConcatTimestepEmbedderND",
    "sgm.modules.encoders.modules.FrozenCLIPEmbedder",
    "sgm.modules.encoders.modules.FrozenOpenCLIPEmbedder2",
    # Used as a no-op conditioner stage by stock configs.
    "torch.nn.Identity",
})


class UnsafeModelTargetError(ValueError):
    """Raised when a model config names a target outside the allowlist."""


def is_model_target_allowed(target, allow_custom=False):
    """Return True when ``target`` may be imported and called for a model.

    ``allow_custom`` is the ``--allow-custom-model-targets`` opt-in: only
    a user who explicitly passes that flag gets arbitrary targets.
    """

    if allow_custom:
        return True

    return isinstance(target, str) and target in ALLOWED_MODEL_TARGETS


def assert_model_target_allowed(target, allow_custom=False):
    """Raise UnsafeModelTargetError unless ``target`` is allowed."""

    if not is_model_target_allowed(target, allow_custom=allow_custom):
        raise UnsafeModelTargetError(
            f"Refusing model target {target!r}: it is not one of the built-in model classes. "
            "A YAML file next to a checkpoint is untrusted input, so only stock model targets are "
            "accepted. If this is your own custom architecture, restart with "
            "--allow-custom-model-targets."
        )


def iter_config_targets(config):
    """Yield every ``target`` value found anywhere inside ``config``.

    Walks plain dicts, OmegaConf DictConfigs (anything with ``.items()``)
    and lists/tuples, so nested targets (UNet, first stage model,
    conditioner, ...) are covered, not just the top-level one. Model
    constructors instantiate nested configs with the same mechanism, so
    checking only the top level would leave the hole open.
    """

    if isinstance(config, dict):
        items = config.items()
    elif hasattr(config, "items"):
        try:
            items = config.items()
        except Exception:
            return
    elif isinstance(config, (list, tuple)):
        for value in config:
            yield from iter_config_targets(value)
        return
    else:
        return

    for key, value in items:
        if key == "target":
            yield value
        else:
            yield from iter_config_targets(value)


def validate_model_config_targets(model_config, allow_custom=False):
    """Refuse a model config naming any target outside the allowlist.

    ``model_config`` is the ``model`` section of a loaded checkpoint
    config. Every ``target`` in it, top-level and nested, must be allowed
    before anything in the config is instantiated. Call this after
    ``repair_config`` has run, so repaired parameters cannot smuggle
    anything past the check, and so a refused target is never constructed
    with repaired parameters.
    """

    for target in iter_config_targets(model_config):
        assert_model_target_allowed(target, allow_custom=allow_custom)
