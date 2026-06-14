# System overview

```
GenAI/
├── config/           # Example + local settings (local files git-ignored)
├── docs/             # Tutorials and architecture
├── models/           # Checkpoints, LoRAs, FaceFusion weights (git-ignored contents)
├── inputs/           # Your source media (git-ignored)
├── outputs/          # Generated results (git-ignored)
├── logs/             # Setup and runtime logs
├── runtime/
│   ├── comfyui/              # Cloned ComfyUI (git-ignored)
│   ├── facefusion/           # Cloned FaceFusion (git-ignored)
│   ├── environments/
│   │   ├── comfyui/          # ComfyUI Python venv
│   │   └── facefusion/       # FaceFusion Python venv
│   └── facefusion-temp/      # Video frame temp (git-ignored)
├── scripts/
│   ├── windows/      # PowerShell entrypoints
│   ├── macos/        # Bash entrypoints
│   └── lib/          # Shared helpers
├── tests/            # Static validation
├── workflows/        # ComfyUI JSON + FaceFusion job examples
└── upstreams.lock.json
```

## Data flow — ComfyUI

1. Checkpoint in `models/comfyui/checkpoints/`
2. Load workflow JSON in browser UI
3. ComfyUI writes images to its output folder (configure Save Image node prefix)
4. Generation metadata embedded in PNG when using default Save Image

## Data flow — FaceFusion

1. Source face + target image/video in `inputs/facefusion/`
2. Launch FaceFusion UI or CLI
3. Processed output in `outputs/facefusion/`
4. Temp frames in `runtime/facefusion-temp/` during video jobs

## Isolation guarantees

| Concern | ComfyUI | FaceFusion |
|---------|---------|------------|
| Python venv | `runtime/environments/comfyui` | `runtime/environments/facefusion` |
| Upstream clone | `runtime/comfyui` | `runtime/facefusion` |
| Default port | 8188 | 7861 (Gradio) |
| GPU stack | PyTorch CUDA / MPS | ONNX Runtime providers |

Applications do not require each other to be running.

## Configuration files

| File | In Git | Purpose |
|------|--------|---------|
| `config/settings.example.env` | Yes | Template for host, ports, profile |
| `config/settings.env` | No | Your overrides |
| `config/local-paths.example.json` | Yes | Directory layout |
| `config/local-paths.json` | No | Your path overrides |
| `upstreams.lock.json` | Yes | Pinned upstream commits |
