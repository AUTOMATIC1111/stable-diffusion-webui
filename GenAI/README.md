# GenAI

Local generative-AI tooling for **ComfyUI** (Stable Diffusion workflows) and **FaceFusion** (image/video face swapping), isolated from the parent AUTOMATIC1111 WebUI in this repository.

## What this project provides

| Component | Purpose | Upstream pin |
|-----------|---------|--------------|
| ComfyUI | Stable Diffusion workflow interface | [v0.9.2](https://github.com/comfyanonymous/ComfyUI/releases/tag/v0.9.2) |
| FaceFusion | Authorized face swap (image + video) | [3.6.1](https://github.com/facefusion/facefusion/releases/tag/3.6.1) |

- **Separate Python environments** for ComfyUI and FaceFusion
- **Windows (RTX 4090)** and **macOS (Apple Silicon)** scripts
- **Pinned upstream commits** in `upstreams.lock.json`
- **Tutorials**, starter workflows, diagnostics, and repair tooling

FaceFusion is the **only** supported face-swapping application in this project. Roop and Roop-derived extensions are intentionally excluded.

## Quick start (Windows + NVIDIA)

```powershell
cd GenAI
.\scripts\windows\setup-all.ps1
.\scripts\windows\doctor.ps1
.\scripts\windows\launch-comfyui.ps1
```

Place a checkpoint in `models/comfyui/checkpoints/` (see [models/README.md](models/README.md)), then load `workflows/comfyui/beginner-text-to-image.json` in ComfyUI.

## Quick start (macOS Apple Silicon)

```bash
cd GenAI
chmod +x scripts/macos/*.sh scripts/lib/common.sh
./scripts/macos/setup-all.sh
./scripts/macos/doctor.sh
./scripts/macos/launch-comfyui.sh
```

## Configuration

1. Copy `config/settings.example.env` → `config/settings.env` (auto-created on first setup)
2. Copy `config/local-paths.example.json` → `config/local-paths.json`
3. Default bind address: **127.0.0.1** (localhost only)

## Scripts

| Action | Windows | macOS |
|--------|---------|-------|
| Setup all | `setup-all.ps1` | `setup-all.sh` |
| Launch ComfyUI | `launch-comfyui.ps1` | `launch-comfyui.sh` |
| Launch FaceFusion | `launch-facefusion.ps1` | `launch-facefusion.sh` |
| Diagnostics | `doctor.ps1` | `doctor.sh` |
| Smoke tests | `smoke-test.ps1` | `smoke-test.sh` |
| Repair envs | `repair.ps1` | `repair.sh` |
| Reset runtime | `reset-runtime.ps1 -Confirm` | `reset-runtime.sh --confirm` |

## Documentation

Start at [docs/00-start-here.md](docs/00-start-here.md).

## Privacy and lawful use

Use only consenting subjects, licensed media, or synthetic fixtures. See [docs/facefusion/10-consent-provenance-and-safety.md](docs/facefusion/10-consent-provenance-and-safety.md).

## Relationship to parent repo

This `GenAI/` tree is **additive** and self-contained. It does not require changes to AUTOMATIC1111 WebUI files (`webui-user.bat`, `modules/`, etc.). If your working tree has unrelated local edits to those files, keep them separate from GenAI commits.

**Port note:** FaceFusion defaults to **7861** to avoid conflicting with the parent WebUI default (**7860**).
