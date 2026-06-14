# ADR-003: Environment Isolation

## Status
Accepted

## Decision
Use **Python `venv`** per application, managed by platform setup scripts.

| Application | Environment path | Python |
|-------------|------------------|--------|
| ComfyUI | `runtime/environments/comfyui/` | 3.10–3.12 |
| FaceFusion | `runtime/environments/facefusion/` | 3.10–3.12 |

## Why venv
- Matches parent repository convention (`venv/` for A1111)
- No conda license or cross-platform conda path complexity
- Officially supported by Python; works on Windows and macOS
- Clear separation: two directories, zero shared `site-packages`

## Incompatible packages (must not share one env)
- **PyTorch** (ComfyUI) vs **ONNX Runtime GPU** (FaceFusion)
- **numpy** version pins differ
- **opencv-python** / **gradio** versions differ
- **CUDA** PyTorch wheels vs **onnxruntime-gpu**

## Rules
1. No `pip install` into system Python
2. No global NumPy/PyTorch/ONNX downgrades
3. Setup scripts install dependencies; launch scripts do **not**
4. Paths are relative to `GenAI/` — no hard-coded user home directories

## Alternatives considered
- **uv**: faster installs; may adopt later without changing isolation model
- **conda**: heavier; unnecessary for two-app layout
- **single venv**: rejected due to numpy/onnx/torch conflicts

## Consequences
- Disk: ~2× Python env overhead (~4–8 GB each with torch)
- Repair: re-run `setup-*.ps1` per app or `repair.ps1`
