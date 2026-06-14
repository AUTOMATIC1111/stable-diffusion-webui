# RTX 4090 configuration

## Verify CUDA PyTorch
After setup:
```powershell
.\runtime\environments\comfyui\Scripts\python.exe -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```
Expected: `cuda True` and `NVIDIA GeForce RTX 4090`.

## If CUDA is False
1. Confirm NVIDIA driver installed (`nvidia-smi`)
2. Re-run `.\scripts\windows\setup-comfyui.ps1`
3. Ensure no CPU-only torch in venv

## Starting recommendations
| Setting | SD1.5 | SDXL |
|---------|-------|------|
| Resolution | 512–768 | 1024 |
| Batch | 1 | 1 |
| Steps | 20–30 | 25–35 |
| CFG | 6–8 | 5–7 |

## FaceFusion on RTX 4090
Set `FACEFUSION_EXECUTION_PROVIDER=cuda` in `config/settings.env`.  
Doctor should list `CUDAExecutionProvider` in ONNX providers after setup.

## Optional performance (not defaults)
- Increase resolution in steps once stable
- Use fp16 checkpoints when available
- Do not enable experimental flags without testing
