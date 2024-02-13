@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--api --nowebui --xformers --no-hashing --no-download-sd-model --timeout-keep-alive 100 --skip-python-version-check --skip-version-check --skip-torch-cuda-test --skip-install --disable-safe-unpickle
call webui.bat
