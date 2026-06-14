# Windows troubleshooting

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `CUDA False` | CPU PyTorch wheel | Re-run `setup-comfyui.ps1` |
| ComfyUI won't start | Port 8188 in use | Change `COMFYUI_PORT` in settings.env |
| Red workflow nodes | Missing checkpoint | Add `.safetensors` to `models/comfyui/checkpoints/` |
| FaceFusion no CUDA | Wrong ONNX package | Re-run `setup-facefusion.ps1` with cuda provider |
| FFmpeg not found | Not on PATH | `winget install ffmpeg`, reopen terminal |
| Black output image | VAE/checkpoint mismatch | Try different VAE or checkpoint |

**First step:** `.\scripts\windows\doctor.ps1`  
**Logs:** `logs/comfyui/`, `logs/facefusion/`, ComfyUI terminal
