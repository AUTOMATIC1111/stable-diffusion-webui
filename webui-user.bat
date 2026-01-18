@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--skip-python-version-check --skip-torch-cuda-test --skip-install
set STABLE_DIFFUSION_REPO=https://github.com/CompVis/stable-diffusion.git
set STABLE_DIFFUSION_COMMIT_HASH=main

call webui.bat
