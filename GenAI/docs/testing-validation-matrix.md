# Validation matrix

Honest test status as of project scaffold (agent environment).

| Component | Static validation | Startup tested | Functional test | Acceleration tested |
|-----------|-------------------|----------------|-----------------|---------------------|
| Config templates | PASS | — | — | — |
| upstreams.lock.json | PASS | — | — | — |
| Workflow JSON | PASS | — | — | — |
| Python validators | PASS | — | — | — |
| PowerShell scripts | PARTIAL (parser) | NOT TESTED | NOT TESTED | NOT TESTED |
| Bash scripts | PARTIAL (review) | NOT TESTED | NOT TESTED | NOT TESTED |
| ComfyUI clone+venv | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| FaceFusion clone+venv | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Image generation | NOT TESTED | NOT TESTED | BLOCKED (no checkpoint) | NOT TESTED |
| Face swap | NOT TESTED | NOT TESTED | BLOCKED (no authorized media) | NOT TESTED |
| RTX 4090 CUDA | NOT TESTED | — | — | NOT TESTED |
| Apple MPS/CoreML | NOT TESTED | — | — | NOT TESTED |

Run static suite:
```powershell
.\scripts\windows\smoke-test.ps1
```
```bash
./scripts/macos/smoke-test.sh
```

Functional tests require user hardware, models, and authorized media.
