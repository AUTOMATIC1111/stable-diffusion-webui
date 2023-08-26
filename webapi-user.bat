@echo off

set PYTHON=E:\Python\Python310\python.exe
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--xformers --no-half --no-half-vae --disable-nan-check --listen --nowebui

call webui.bat
