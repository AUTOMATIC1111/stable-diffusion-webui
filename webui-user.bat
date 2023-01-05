@echo off

set PYTHON=
set GIT=
set VENV_DIR=-
set SAFETENSORS_FAST_GPU=1
set COMMANDLINE_ARGS=--force-enable-xformers --listen --api --disable-safe-unpickle --opt-channelslast --enable-insecure-extension-access --theme dark

call webui.bat
