@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--opt-split-attention  --medvram --api --xformers

call webui.bat
