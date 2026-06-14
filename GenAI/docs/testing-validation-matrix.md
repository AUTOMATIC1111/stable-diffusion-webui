# Validation matrix

Honest test status after Windows runtime validation on RTX 4090 host (2026-06-14). Commit context: `46ddd9ab` (Add cross-platform ComfyUI and FaceFusion environment). Mac rows: **NOT TESTED** on this Windows host.

| Component | Static validation | Startup tested | Functional test | Acceleration tested |
|-----------|-------------------|----------------|-----------------|---------------------|
| Config templates | PASS | PASS (setup created settings) | N/A | N/A |
| upstreams.lock.json | PASS | PASS (pinned clones) | N/A | N/A |
| Workflow JSON | PASS | N/A | NOT TESTED | N/A |
| Python validators | PASS | N/A | N/A | N/A |
| PowerShell scripts | PASS (smoke-test static) | PARTIAL | NOT TESTED | NOT TESTED |
| Bash scripts | NOT TESTED (Windows host) | NOT TESTED | NOT TESTED | NOT TESTED |
| ComfyUI clone+venv | PASS | PASS (HTTP 200 @8188)* | BLOCKED (no checkpoint) | PASS (CUDA RTX 4090) |
| FaceFusion clone+venv | PASS | NOT TESTED (no authorized media) | BLOCKED (no authorized media) | PASS (CUDA/TRT ONNX providers) |
| Image generation | NOT TESTED | N/A | BLOCKED (no checkpoint) | NOT TESTED |
| Face swap | NOT TESTED | N/A | BLOCKED (no authorized media in inputs/facefusion) | NOT TESTED |
| RTX 4090 CUDA | PASS (doctor + torch) | N/A | N/A | PASS (ComfyUI cuda=True; ONNX CUDA) |
| Apple MPS/CoreML | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |

* ComfyUI v0.9.2 at pinned commit omits `requests` from `requirements.txt`; first validation run needed manual `pip install requests` before HTTP 200 (~15s). `setup-comfyui.ps1` now installs `requests` after requirements on fresh setup.

## Windows validation notes

- **setup-all.ps1**: First run failed on PowerShell treating `git` stderr as terminating (`Invoke-GitClonePinned`); patched `GenAI-Common.ps1` to set `$ErrorActionPreference = 'Continue'` during git ops. Subsequent setup-comfyui + setup-facefusion completed successfully.
- **doctor.ps1**: Failed parse until em-dash characters replaced with ASCII hyphen (Windows encoding). After fix: all PASS except **FFmpeg WARN**.
- **launch-comfyui.ps1 / launch-facefusion.ps1**: Same em-dash parse issue fixed for startup smoke.
- **GENAI_PROFILE**: `balanced` adds no extra args; `compatibility` → `--disable-smart-memory`; `performance` → `--highvram` (matches launch-comfyui.ps1 switch).
- **models/comfyui/checkpoints/**: empty (no download performed).
- **inputs/facefusion/**: empty (no face swap run).

Run static suite:
```powershell
.\scripts\windows\smoke-test.ps1
```
```bash
./scripts/macos/smoke-test.sh
```

Functional tests require user hardware, models, and authorized media.
