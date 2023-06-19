@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--autolaunch  --no-gradio-queue  --allow-code --no-half-vae --no-half --api  --xformers  --skip-version-check --skip-torch-cuda-test 
git pull


call webui.bat
