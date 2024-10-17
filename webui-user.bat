
@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--xformers  --api --autolaunch --opt-split-attention --no-half-vae --theme dark

call webui.bat
