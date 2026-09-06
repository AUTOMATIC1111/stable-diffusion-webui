# Installation Guide

This guide installs the web UI with Python 3.12 in an isolated virtual environment. Python 3.10 and 3.11 remain supported for existing installations.

## Windows 10/11 with an NVIDIA GPU

1. Install [64-bit Python 3.12](https://www.python.org/downloads/) and select **Add Python to PATH** in the installer.
2. Install [Git for Windows](https://git-scm.com/download/win).
3. Open PowerShell and clone the repository:

   ```powershell
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   Set-Location stable-diffusion-webui
   ```

4. Start the web UI:

   ```powershell
   .\webui-user.bat
   ```

The launcher creates `venv` automatically, installs Python 3.12-compatible Torch 2.4.1 and torchvision 0.19.1 wheels, then installs the remaining requirements. Open the URL printed in the terminal, normally `http://127.0.0.1:7860`.

To use a different Python installation, set it in `webui-user.bat` before launching:

```bat
set PYTHON=C:\Path\To\Python312\python.exe
```

If a previous environment was created with another Python version, remove only the repository's `venv` directory and run `webui-user.bat` again.

## Linux

Install the system packages first. On Debian or Ubuntu:

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv python3.12-dev libgl1 libglib2.0-0
```

Clone and launch:

```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
sed -i 's/^#python_cmd="python3"/python_cmd="python3.12"/' webui-user.sh
./webui.sh
```

The managed launcher creates `venv`, installs the dependencies, and starts the web UI.

## GPU notes

- NVIDIA users should install a current NVIDIA driver. The launcher uses the CUDA 12.1 PyTorch index by default.
- AMD, Intel, Apple Silicon, CPU-only, and NPU installations need the platform-specific instructions linked from the [README](README.md#installation-and-running).
- Checkpoint files are not included. Place them under `models/Stable-diffusion/` before generating images.

## Troubleshooting

- If dependencies were installed under another Python version, delete `venv` and launch again with Python 3.12.
- If Torch reports that CUDA is unavailable, verify the NVIDIA driver and use `--skip-torch-cuda-test` only when CPU or another accelerator is intentional.
- Do not run the launcher as Administrator unless the installation directory requires it.