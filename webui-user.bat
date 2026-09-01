@echo off

set PYTHON="C:\Users\hellc\OneDrive\Microsoft Copilot Chat Files\Documents\stable-diffusion-webui-a1111-patched\venv\Scripts\python.exe"
set GIT="C:\Users\hellc\Downloads\Git-2.55.0.5-64-bit.exe"
set VENV_DIR="C:\Users\hellc\OneDrive\Microsoft Copilot Chat Files\Documents\stable-diffusion-webui-a1111-patched\venv"
set COMMANDLINE_ARGS=--low-vram

call webui.bat

set STABLE_DIFFUSION_REPO=https://github.com/w-e-w/stablediffusion.git
