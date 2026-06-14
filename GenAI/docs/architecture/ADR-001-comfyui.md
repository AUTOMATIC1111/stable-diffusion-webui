# ADR-001: ComfyUI as the Stable Diffusion Interface

## Status
Accepted

## Context
This project needs a local Stable Diffusion workflow engine that supports node-based pipelines, reproducible JSON workflows, and cross-platform GPU acceleration without duplicating the parent repo's AUTOMATIC1111 WebUI.

## Decision
Use **ComfyUI** ([comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)) with a **pinned manual installation** cloned into `runtime/comfyui/`.

### Why ComfyUI
- Node graph maps directly to diffusion pipeline stages (model load, encode, sample, decode)
- Workflow JSON is portable and version-controllable
- Active upstream, broad model support (SD1.5, SDXL, Flux, etc.)
- Official support for NVIDIA CUDA and Apple MPS

### Manual vs desktop installer
**Manual clone + project scripts** was chosen over ComfyUI Desktop because:
- Reproducible pinned commits in `upstreams.lock.json`
- Scriptable setup, doctor, and repair
- Consistent layout with FaceFusion bootstrap
- No dependency on a separate desktop auto-updater

### Platform support
| Platform | Backend | PyTorch install |
|----------|---------|-----------------|
| Windows + RTX 4090 | CUDA | `cu124` index from download.pytorch.org |
| Apple Silicon | MPS | Default PyTorch wheel from PyPI |
| Intel Mac / CPU | CPU | Documented fallback; not primary target |

### Python and PyTorch
- Python **3.10–3.12** per ComfyUI v0.9.2 requirements
- Install **torch/torchvision/torchaudio** before `requirements.txt`
- Windows: verify `torch.cuda.is_available()` after setup

### Update strategy
Explicit edit of `upstreams.lock.json` + re-run setup. No silent tracking of `master`.

### Model directory strategy
Models live under `models/comfyui/` with subfolders (`checkpoints`, `vae`, `loras`, etc.). `extra_model_paths.yaml` in the ComfyUI clone points to this tree. Optional external storage via junction/symlink documented in `models/README.md`.

### Workflow portability
Committed workflows use stock ComfyUI nodes only. Missing custom nodes are documented, not auto-installed.

### Custom-node policy
No custom nodes in the default install. Add only after explicit review (supply chain, license, maintenance).

## Consequences
- Users learn ComfyUI node graphs (see tutorials)
- A1111-specific extensions do not apply here
- ComfyUI runs independently of FaceFusion
