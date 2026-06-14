# Apple Silicon installation

## Prerequisites
```bash
xcode-select --install   # if needed
brew install python@3.12 git ffmpeg
```

## Install
```bash
cd GenAI
chmod +x scripts/macos/*.sh scripts/lib/common.sh
./scripts/macos/setup-all.sh
./scripts/macos/doctor.sh
```

ComfyUI uses **PyTorch MPS** when available. FaceFusion uses **coreml** or **cpu** per `FACEFUSION_EXECUTION_PROVIDER` in `config/settings.env`.

## Verify MPS
```bash
./runtime/environments/comfyui/bin/python -c "import torch; print('mps', torch.backends.mps.is_available())"
```

## Unified memory
Close heavy apps before SDXL or long FaceFusion video jobs — memory is shared between CPU and GPU.
