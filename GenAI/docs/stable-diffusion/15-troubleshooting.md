# Stable Diffusion troubleshooting

| Symptom | Cause | Action |
|---------|-------|--------|
| Black image | VAE mismatch | Try external VAE or different checkpoint |
| CUDA OOM | Resolution too high | Lower size or batch |
| Red nodes | Missing node type | Use stock workflow or install reviewed custom node |
| Slow every gen | CPU mode | Re-run setup; verify CUDA/MPS |
| Wrong model | Filename mismatch | Fix CheckpointLoader widget |

**Diagnostic:** `doctor.ps1` / `doctor.sh`  
**Logs:** `logs/comfyui/`, ComfyUI terminal output
