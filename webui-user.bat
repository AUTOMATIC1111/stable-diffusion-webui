@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--xformers --lowvram --precision full --no-half --skip-torch-cuda-test

call webui.bat
