@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= --skip-torch-cuda-test --precision full --no-half

call webui.bat
