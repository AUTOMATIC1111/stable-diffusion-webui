# RTX 5090 Compatibility Patch Notes

## Summary
This branch adds compatibility guards and dependency fixes so AUTOMATIC1111 WebUI can run with modern RTX 5090 + PyTorch nightly (`cu128`) while using CompVis Stable Diffusion repository layout.

## Code Changes

### 1) Guard missing `ATTENTION_MODES`
- File: `modules/sd_hijack.py`
- Change: wrap assignment to `BasicTransformerBlock.ATTENTION_MODES` with `hasattr(...)`.
- Reason: some `ldm` variants do not define this attribute.

### 2) Guard missing `use_linear` on `SpatialTransformer`
- File: `modules/sd_hijack_unet.py`
- Change: replace direct `self.use_linear` access with:
  - `use_linear = getattr(self, "use_linear", False)`
- Reason: CompVis transformer blocks may not expose `use_linear`.

### 3) Make depth class import optional
- File: `modules/processing.py`
- Change: `LatentDepth2ImageDiffusion` import wrapped in `try/except ImportError`.
- Change: `isinstance(...)` check now guarded by `LatentDepth2ImageDiffusion is not None`.
- Reason: class may not exist in all `ldm` implementations.

### 4) Make `AddMiDaS` import optional
- File: `modules/processing.py`
- Change: `AddMiDaS` import wrapped in `try/except` with a no-op fallback class.
- Reason: `ldm.data.util` is absent in some CompVis repo states.

### 5) Make MiDaS module optional
- File: `modules/sd_models.py`
- Change: wrap `import ldm.modules.midas as midas` in `try/except`.
- Change: `enable_midas_autodownload()` returns early when `midas is None`.
- Reason: prevents startup crash when MiDaS module is unavailable.

### 6) Update default Stable Diffusion upstream for working clone path
- File: `modules/launch_utils.py`
- Change: `STABLE_DIFFUSION_REPO` default set to `https://github.com/CompVis/stable-diffusion.git`.
- Change: default `STABLE_DIFFUSION_COMMIT_HASH` set to `21f890f9da3cfbeaba8e2ac3c425ee9e998d5229`.
- Reason: previous default URL may be unreachable in current environment.

## Environment / Install Actions
- Installed GitHub CLI (`gh`) in user path: `~/.local/bin/gh`
- Forked upstream: `AUTOMATIC1111/stable-diffusion-webui` -> `PiercingXX/stable-diffusion-webui`
- Cloned into: `/media/Working-Storage/GitHub/stable-diffusion-webui`
- Created branch: `feat/rtx5090-compat`

Python environment:
- Python: `3.10.14`
- Torch: `2.12.0.dev20260310+cu128`
- CUDA available: `True`
- GPU detected: `NVIDIA GeForce RTX 5090`

Installed dependencies:
- `requirements_versions.txt`
- `clip` (OpenAI CLIP zip, `--no-build-isolation`)
- `taming-transformers-rom1504`
- Additional runtime deps used by startup path (`pytorch_lightning`, `kornia`, `diskcache`, etc.)

## Validation Performed

### Runtime checks
- `torch.cuda.is_available()` -> `True`
- `torch.cuda.get_device_name(0)` -> `NVIDIA GeForce RTX 5090`

### WebUI smoke startup
Command used for startup validation:
- `webui.py --skip-python-version-check --listen --server-name 0.0.0.0 --port 7860 --skip-load-model-at-start --no-download-sd-model`

Observed in logs:
- `Running on local URL:  http://0.0.0.0:7860`
- `Startup time: 4.2s ...`

HTTP response verified:
- `curl http://0.0.0.0:7860/` returned HTML.

## Notes / Remaining Validation
- Full text-to-image generation requires a model checkpoint to be present or fully downloaded.
- This branch has been validated for startup/runtime compatibility and serving UI on RTX 5090 stack.
