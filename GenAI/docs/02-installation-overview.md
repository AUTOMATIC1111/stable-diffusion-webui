# Installation overview

GenAI uses **bootstrap scripts** that clone pinned upstream releases and create **isolated Python virtual environments**. Models and private media are never downloaded automatically without your action.

## Order of operations

1. **Install OS prerequisites** (Git, Python, FFmpeg, GPU drivers)
2. **Run `setup-all`** — clones ComfyUI + FaceFusion, creates venvs, installs dependencies
3. **Run `doctor`** — verifies GPU, envs, ports, FFmpeg
4. **Configure paths** (optional) — edit `config/settings.env` and `config/local-paths.json`
5. **Add models** — manual download; see [models/README.md](../models/README.md)
6. **Run smoke tests** — static validation without requiring models

## Platform guides

- Windows + RTX 4090: [windows/installation.md](windows/installation.md)
- macOS Apple Silicon: [macos/apple-silicon-installation.md](macos/apple-silicon-installation.md)
- Intel Mac: [macos/intel-mac-assessment.md](macos/intel-mac-assessment.md)

## What setup does *not* do

- Download multi-GB Stable Diffusion checkpoints (license + size)
- Download FaceFusion ONNX models until first run or explicit `facefusion.py force-download`
- Modify the parent A1111 `venv/` or `webui-user.bat`
- Install Roop or ComfyUI custom nodes

## Idempotency

Re-running `setup-comfyui` or `setup-facefusion` is safe: existing clones are checked out to the pinned commit; venvs are reused and dependencies refreshed.

## Repair vs reset

| Command | Deletes models? | Deletes outputs? | Use when |
|---------|-----------------|------------------|----------|
| `repair` | No | No | Broken venv, failed pip install |
| `reset-runtime -Confirm` | No | No | Corrupt clone; wipes runtime + venv only |

## Verification

```powershell
# Windows
.\scripts\windows\doctor.ps1
.\scripts\windows\smoke-test.ps1
```

```bash
# macOS
./scripts/macos/doctor.sh
./scripts/macos/smoke-test.sh
```

Expected: all static checks **PASS**; GPU checks **PASS** or **WARN** until setup completes on target hardware.
