# Windows installation

## Prerequisites
1. Windows 10/11 64-bit
2. [Git for Windows](https://git-scm.com/download/win)
3. Python 3.10–3.12 (check **Add to PATH** during install)
4. [FFmpeg](https://ffmpeg.org/download.html) on PATH (`winget install ffmpeg`)
5. Latest [NVIDIA driver](https://www.nvidia.com/drivers) for RTX 4090

## Install
```powershell
cd GenAI
.\scripts\windows\setup-all.ps1
.\scripts\windows\doctor.ps1
```

Setup clones ComfyUI **v0.9.2** and FaceFusion **3.6.1**, creates separate venvs, installs PyTorch **cu124** for ComfyUI, and configures FaceFusion ONNX provider per `settings.env`.

## Optional profiles
Edit `config/settings.env`:
- `GENAI_PROFILE=compatibility|balanced|performance`
- `FACEFUSION_EXECUTION_PROVIDER=cuda|directml|cpu`

## Next steps
- [RTX 4090 tuning](rtx-4090-configuration.md)
- [First image](../stable-diffusion/02-first-image.md)
