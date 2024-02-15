@echo off

set PYTHON=
set GIT=
set VENV_DIR=
<<<<<<< HEAD
set COMMANDLINE_ARGS= --api
=======
set COMMANDLINE_ARGS= --api --xformers --no-hashing --no-download-sd-model --timeout-keep-alive 100 --skip-python-version-check --skip-version-check --skip-torch-cuda-test --skip-install
>>>>>>> e975c8c1f7f314b74a29b0df19cc7a61fcb23b0f
call webui.bat