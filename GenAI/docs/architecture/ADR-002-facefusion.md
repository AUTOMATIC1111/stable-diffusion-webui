# ADR-002: FaceFusion as the Face-Swapping Application

## Status
Accepted

## Context
The project requires local image and video face manipulation with browser UI, CLI, and headless modes.

## Decision
Use official **[FaceFusion](https://github.com/facefusion/facefusion)** pinned at **3.6.1**, cloned to `runtime/facefusion/`.

### Why FaceFusion
- Active upstream with image + video pipeline, enhancement, and job system
- Documented execution providers and CLI (`facefusion.py`)
- Gradio UI for interactive use

### Why Roop is excluded
- Roop is unmaintained; community forks vary in quality and licensing
- Roop-derived Stable Diffusion extensions conflate face swap with the diffusion stack
- FaceFusion is the designated supported tool; one supported path reduces support burden

### Why separate from ComfyUI
FaceFusion pins **numpy 2.2.1**, **onnxruntime 1.24.4**, and **gradio 5.x** — incompatible with ComfyUI's PyTorch-centric stack. Separate venvs prevent dependency conflicts.

### Execution providers

| OS | Preferred | Fallback | Notes |
|----|-----------|----------|-------|
| Windows + NVIDIA | `cuda` (onnxruntime-gpu) | `directml`, `cpu` | Verify with `ort.get_available_providers()` |
| macOS Apple Silicon | `coreml` (onnxruntime-silicon when available) | `cpu` | CoreML support varies by model/op |
| Intel Mac | `cpu` | — | Not a primary target |

**Do not claim CUDA/CoreML is active without runtime verification.**

### FFmpeg
Required for video encode/decode. Must be on PATH (`ffmpeg`, `ffprobe`).

### ONNX Runtime
Installed per platform in the FaceFusion venv only — never globally replaced.

### Environment isolation
Dedicated venv at `runtime/environments/facefusion/`. No shared site-packages with ComfyUI.

### Update strategy
Same as ComfyUI: pin in lock file, manual upgrade after reading release notes.

### Model download
FaceFusion downloads models on first use via built-in commands. Document manual steps; do not commit `.onnx` weights.

### Privacy
- Source faces stay in git-ignored `inputs/facefusion/`
- No upload to external services by default
- Localhost binding only unless user changes `GENAI_HOST`

## Consequences
- Users run FaceFusion via `launch-facefusion.ps1` / `.sh`, not inside ComfyUI
- ReActor and similar ComfyUI nodes are not installed by default
