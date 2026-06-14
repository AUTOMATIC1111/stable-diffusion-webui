# Model management

AI model weights are **never committed to Git**. This directory holds local files only.

## Directory layout

```
models/
├── comfyui/
│   ├── checkpoints/     # .safetensors / .ckpt base models
│   ├── vae/             # Optional external VAE
│   ├── loras/           # LoRA adapters
│   ├── embeddings/      # Textual inversion
│   ├── controlnet/      # ControlNet weights
│   └── upscale_models/  # ESRGAN, etc.
└── facefusion/          # FaceFusion-downloaded ONNX (or symlink)
```

ComfyUI resolves paths via `extra_model_paths.yaml` generated during setup.

## Checkpoints

| Family | Typical file | VRAM (512px) | Notes |
|--------|--------------|--------------|-------|
| SD 1.5 | `v1-5-pruned-emaonly.safetensors` | ~4 GB | Starter tutorial default |
| SDXL | `sd_xl_base_1.0.safetensors` | ~8 GB | 1024px native |
| Flux | varies | 12+ GB | Check ComfyUI node support |

**Prefer `.safetensors`** over `.ckpt` (no arbitrary pickle code). Safetensors are not inherently trustworthy — verify source and hash.

## Obtaining models

1. Read the model card license (CreativeML Open RAIL-M, etc.)
2. Accept gating on Hugging Face if required — **manual login in browser**
3. Download to the correct subfolder
4. Record in `config/model-manifest.json` (copy from `model-manifest.example.json`)

## External shared storage

### Windows junction (same volume)
```powershell
New-Item -ItemType Junction -Path "models\comfyui\checkpoints" -Target "D:\AI\Models\checkpoints"
```

### macOS symlink
```bash
ln -s ~/AI/Models/checkpoints models/comfyui/checkpoints
```

## FaceFusion models

FaceFusion downloads required ONNX models on first use:

```bash
cd runtime/facefusion
../environments/facefusion/bin/python facefusion.py force-download
```

(Windows: use `Scripts\python.exe`.)

## Security

- Download only from official repos or trusted mirrors
- Compare SHA256 when published
- Do not run checkpoint files as scripts
- Scan unfamiliar models in isolated environment

## Storage planning

| Content | Typical size |
|---------|--------------|
| SD 1.5 checkpoint | ~4 GB |
| SDXL base | ~6.5 GB |
| LoRA collection | 10–100+ GB |
| FaceFusion models | ~2–5 GB |

Plan **50 GB minimum** free space for comfortable local work.
