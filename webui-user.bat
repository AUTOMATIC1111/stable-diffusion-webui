@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--medvram --precision full --no-half --unload-gfpgan --opt-split-attention

call webui.bat
