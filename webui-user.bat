REM rm *.json
git pull
for /D %%i in (extensions\*) do @ git -C "%%i" pull --recurse-submodules
@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--config repositories/stable-diffusion/configs/stable-diffusion/v2-inference-v.yaml
set STABLE_DIFFUSION_REPO=https://github.com/Stability-AI/stablediffusion
set STABLE_DIFFUSION_COMMIT_HASH=33910c386eaba78b7247ce84f313de0f2c314f61

call webui.bat
